#!/usr/bin/env python3
# tools/rebuild_graph_knn.py — HX-AM v4.5.1
"""
Правильная пересборка invariant_graph.json через k-NN.

ПОЧЕМУ ПОРОГОВЫЙ ПОДХОД НЕ РАБОТАЕТ:
  При 153 узлах и math-перенасыщенном архиве среднее cosine similarity
  между гипотезами ≈ 0.60-0.75. Любой порог ниже 0.80 даёт тысячи рёбер.
  Порог 0.65 → 6000+ рёбер (55% плотность = почти полная клика).

ПРАВИЛЬНОЕ РЕШЕНИЕ — k-NN граф:
  Каждый узел соединяется только с топ-K ближайшими соседями.
  При K=8 и 153 узлах максимум рёбер = 153×8/2 = 612.
  Дополнительный фильтр sim_floor отсекает случайные связи.

Формула веса ребра (оригинальная из invariant_engine.py):
  weight = similarity × (1 + domain_distance) × avg_specificity

Запуск из корня проекта (Linux/WSL):
  python tools/rebuild_graph_knn.py
  python tools/rebuild_graph_knn.py --k 10
  python tools/rebuild_graph_knn.py --k 6 --floor 0.65
  python tools/rebuild_graph_knn.py --dry-run
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rebuild_knn")

PROJECT_ROOT   = Path(__file__).parent.parent
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
SEM_INDEX_PATH = ARTIFACTS_DIR / "semantic_index.jsonl"
GRAPH_PATH     = ARTIFACTS_DIR / "invariant_graph.json"

_DOMAIN_MAP = {
    "социология": "sociology", "психология": "psychology",
    "физика": "physics", "биология": "biology",
    "математика": "mathematics", "химия": "chemistry",
    "лингвистика": "linguistics", "экономика": "economics",
    "экология": "ecology", "нейронаука": "neuroscience",
    "геология": "geology", "медицина": "medicine",
    "астрономия": "astronomy", "social": "sociology",
    "psych": "psychology", "neuro": "neuroscience",
    "bio": "biology", "chem": "chemistry", "math": "mathematics",
    "econ": "economics",
}

def norm_domain(d):
    d = (d or "general").strip().lower()
    return _DOMAIN_MAP.get(d, d)


def load_artifacts():
    result = {}
    if not ARTIFACTS_DIR.exists():
        return result
    for f in ARTIFACTS_DIR.glob("*.json"):
        if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
            continue
        try:
            art    = json.loads(f.read_text(encoding="utf-8"))
            art_id = art.get("id", f.stem)
            data   = art.get("data", {})
            gen    = data.get("gen", {})
            struct = data.get("structural", {})
            arch   = art.get("archivist") or {}
            sim_d  = art.get("simulation") or {}
            result[art_id] = {
                "domain":          norm_domain(data.get("domain") or gen.get("domain") or "general"),
                "b_sync":          float(gen.get("b_sync") or 0.5),
                "specificity":     float(struct.get("specificity") or 0.5),
                "stability":       struct.get("stability") or "unknown",
                "artifact_type":   struct.get("artifact_type") or "unknown",
                "survival":        (struct.get("translation") or {}).get("survival", "UNKNOWN"),
                "novelty":         (arch.get("novelty") or "").split(":")[0],
                "novelty_score":   float(arch.get("novelty_score") or 0.5),
                "stability_score": float(sim_d.get("stability_score") or 0.0),
                "has_four_d":      bool(struct.get("has_four_d") or gen.get("four_d_matrix")),
                "stress_stable":   struct.get("stress_stable"),
                "linked_to":       [l for l in (arch.get("linked_to") or []) if l != art_id],
                "suggested_tags":  arch.get("suggested_tags") or [],
            }
        except Exception as e:
            logger.warning(f"  skip {f.name}: {e}")
    return result


def load_embeddings():
    if not SEM_INDEX_PATH.exists():
        logger.error(f"  semantic_index.jsonl не найден: {SEM_INDEX_PATH}")
        sys.exit(1)
    logger.info("  Загружаем sentence-transformers (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.error("  Установи: pip install sentence-transformers")
        sys.exit(1)

    entries = []
    with open(SEM_INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    logger.info(f"  Encoding {len(entries)} гипотез...")
    texts = [e["invariant"] for e in entries]
    vecs  = model.encode(texts, show_progress_bar=False, batch_size=64)

    result = {}
    for e, vec in zip(entries, vecs):
        eid = e.get("id")
        if eid:
            result[eid] = {"vec": vec, "domain": norm_domain(e.get("domain", "general"))}
    logger.info(f"  Загружено {len(result)} эмбеддингов")
    return result, model


def build_knn_graph(artifacts, embeddings, st_model, k, sim_floor):
    valid_ids = sorted(set(artifacts.keys()) & set(embeddings.keys()))
    logger.info(f"  Узлов: {len(valid_ids)}")

    vecs  = np.array([embeddings[nid]["vec"] for nid in valid_ids], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    vecs_n = vecs / norms
    sim_m  = vecs_n @ vecs_n.T
    np.fill_diagonal(sim_m, -1.0)

    sims_pos = sim_m[sim_m > 0]
    logger.info(f"  Similarity: mean={float(sims_pos.mean()):.3f} "
                f"p75={float(np.percentile(sims_pos, 75)):.3f} "
                f"p90={float(np.percentile(sims_pos, 90)):.3f}")

    dom_cache = {}
    def get_dist(d1, d2):
        key = tuple(sorted([d1, d2]))
        if key not in dom_cache:
            try:
                v1 = st_model.encode(d1)
                v2 = st_model.encode(d2)
                dom_cache[key] = round(float(scipy_cosine(v1, v2)), 3)
            except Exception:
                dom_cache[key] = 0.0
        return dom_cache[key]

    edges = {}
    for i, nid_i in enumerate(valid_ids):
        sims_i    = sim_m[i].copy()
        top_k_idx = np.argsort(sims_i)[::-1][:k]
        for j in top_k_idx:
            sim = float(sims_i[j])
            if sim < sim_floor:
                continue
            nid_j = valid_ids[j]
            key   = (min(i, j), max(i, j))
            if key in edges:
                continue
            dom_i    = embeddings[nid_i]["domain"]
            dom_j    = embeddings[nid_j]["domain"]
            dist     = get_dist(dom_i, dom_j)
            spec_i   = artifacts[nid_i].get("specificity", 0.5)
            spec_j   = artifacts[nid_j].get("specificity", 0.5)
            avg_spec = (spec_i + spec_j) / 2
            weight   = round(sim * (1 + dist) * avg_spec, 4)
            edges[key] = {
                "source":           nid_i,
                "target":           nid_j,
                "similarity":       round(sim, 4),
                "domain_distance":  dist,
                "specificity":      round(avg_spec, 4),
                "four_d_resonance": 0.0,
                "weight":           weight,
            }

    links = list(edges.values())
    logger.info(f"  Рёбер: {len(links)}")

    dom_counts = {}
    for nid in valid_ids:
        d = artifacts.get(nid, {}).get("domain", "general")
        dom_counts[d] = dom_counts.get(d, 0) + 1

    nodes = []
    for nid in valid_ids:
        a = artifacts.get(nid, {})
        nodes.append({
            "id":              nid,
            "domain":          a.get("domain", "general"),
            "b_sync":          a.get("b_sync", 0.5),
            "stability":       a.get("stability", "unknown"),
            "artifact_type":   a.get("artifact_type", "unknown"),
            "specificity":     a.get("specificity", 0.5),
            "survival":        a.get("survival", "UNKNOWN"),
            "novelty":         a.get("novelty", ""),
            "novelty_score":   a.get("novelty_score", 0.5),
            "stability_score": a.get("stability_score", 0.0),
            "has_four_d":      a.get("has_four_d", False),
            "stress_stable":   a.get("stress_stable"),
            "linked_to":       a.get("linked_to", []),
            "suggested_tags":  a.get("suggested_tags", []),
        })

    return {
        "directed":       False,
        "multigraph":     False,
        "graph":          {},
        "nodes":          nodes,
        "links":          links,
        "_rebuilt_at":    datetime.now(timezone.utc).isoformat(),
        "_rebuild_k":     k,
        "_rebuild_floor": sim_floor,
        "_node_count":    len(nodes),
        "_edge_count":    len(links),
        "_domain_stats":  dict(sorted(dom_counts.items(), key=lambda x: -x[1])),
    }


def print_stats(graph_data):
    n       = graph_data["_node_count"]
    e       = graph_data["_edge_count"]
    max_e   = n * (n - 1) // 2
    density = e / max_e * 100 if max_e > 0 else 0
    avg_deg = 2 * e / n if n > 0 else 0

    deg = {}
    for link in graph_data["links"]:
        for key in ("source", "target"):
            nid = link[key]
            deg[nid] = deg.get(nid, 0) + 1
    max_deg  = max(deg.values()) if deg else 0
    isolated = sum(1 for node in graph_data["nodes"] if node["id"] not in deg)

    print(f"\n  Узлов:      {n}")
    print(f"  Рёбер:      {e}  (макс возможно {max_e})")
    print(f"  Плотность:  {density:.2f}%  (было 45-55%)")
    print(f"  Avg degree: {avg_deg:.1f}")
    print(f"  Max degree: {max_deg}")
    print(f"  Изолиров.:  {isolated}")
    print()
    print("  Домены:")
    for dom, cnt in sorted(graph_data["_domain_stats"].items(), key=lambda x: -x[1]):
        bar = "█" * min(cnt, 25)
        pct = cnt / n * 100 if n else 0
        print(f"    {dom:20s} {bar} {cnt:3d} ({pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",         type=int,   default=8)
    parser.add_argument("--floor",     type=float, default=0.60)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    print(f"\n🔧 k-NN Graph Rebuild  k={args.k}  floor={args.floor}")
    print(f"   {'DRY RUN' if args.dry_run else 'LIVE'}")

    print("\n📦 Артефакты...")
    artifacts  = load_artifacts()
    print(f"   {len(artifacts)} артефактов")

    print("\n🧠 Эмбеддинги...")
    embeddings, st_model = load_embeddings()

    print(f"\n🕸️  Строим граф...")
    graph_data = build_knn_graph(artifacts, embeddings, st_model, args.k, args.floor)

    print_stats(graph_data)

    if args.dry_run:
        print("\n(dry-run: файл не записан)")
        return

    if GRAPH_PATH.exists() and not args.no_backup:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = GRAPH_PATH.parent / f"invariant_graph.{ts}.bak.json"
        shutil.copy2(GRAPH_PATH, bak)
        print(f"\n💾 Backup: {bak.name}")

    GRAPH_PATH.write_text(
        json.dumps(graph_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ Записано: {GRAPH_PATH}")
    print("   Перезапустите сервер.")


if __name__ == "__main__":
    main()
