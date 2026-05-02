# tools/restore_graph.py — HX-AM v4.2
"""
Восстанавливает invariant_graph.json из артефактов + semantic_index.

ПРОБЛЕМА КОТОРУЮ РЕШАЕТ:
  rebuild_graph_clean.py создал граф из 4D векторов:
  - 94.5% пар имеют косинус > 0.7 → 5160 рёбер → клубок
  - Узлы содержат только domain/stability_score/math_model → серые

РЕШЕНИЕ:
  1. Читает артефакты → правильные атрибуты узлов
  2. Читает semantic_index.jsonl → text embeddings
  3. Рёбра только при text cosine sim > 0.65 (как invariant_engine)
  4. Вес = similarity × (1 + domain_distance) × avg_specificity

CLI:
  python tools/restore_graph.py
  python tools/restore_graph.py --dry-run
  python tools/restore_graph.py --threshold 0.70
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("restore_graph")

ARTIFACTS_DIR   = Path("artifacts")
GRAPH_PATH      = Path("artifacts/invariant_graph.json")
SEM_INDEX_PATH  = Path("artifacts/semantic_index.jsonl")
DEFAULT_THRESHOLD = 0.65

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
    "econ": "economics", "sociolinguistics": "linguistics",
}

def norm_domain(d):
    d = (d or "general").strip().lower()
    return _DOMAIN_MAP.get(d, d)

def _infer_type(stability, struct):
    phase = (struct.get("phase_signal") or {}).get("signal", "")
    if phase == "sigma_primitive_candidate":
        return "sigma_primitive_candidate"
    if struct.get("is_bridge"):
        return "hyx-portal"
    m = {"stable_cluster": "hyx-artifact", "weak_pattern": "weak_pattern",
         "mixed_patterns": "weak_pattern", "noise": "noise", "isolated": "noise"}
    return m.get(stability, stability or "unknown")

def load_artifacts():
    items = []
    for f in sorted(ARTIFACTS_DIR.glob("*.json")):
        if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
            continue
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        art_id = art.get("id", f.stem)
        data   = art.get("data", {})
        gen    = data.get("gen", {})
        ver    = data.get("ver", {})
        struct = data.get("structural", {})
        arch   = art.get("archivist") or {}
        sim_d  = art.get("simulation") or {}

        domain      = norm_domain(data.get("domain") or gen.get("domain") or "general")
        b_sync      = float(gen.get("b_sync") or 0.5)
        translation = struct.get("translation") or ver.get("translation") or {}
        survival    = translation.get("survival", "UNKNOWN") if isinstance(translation, dict) else "UNKNOWN"
        stability   = struct.get("stability") or "unknown"
        art_type    = struct.get("artifact_type") or _infer_type(stability, struct)
        specificity = float(struct.get("specificity") or 0.5)
        novelty     = (arch.get("novelty") or "?").split(":")[0]

        items.append({
            "id":              art_id,
            "domain":          domain,
            "b_sync":          b_sync,
            "stability":       stability,
            "artifact_type":   art_type,
            "specificity":     specificity,
            "survival":        survival,
            "novelty":         novelty,
            "novelty_score":   float(arch.get("novelty_score") or 0.5),
            "math_verification": arch.get("mathematical_verification") or "TERMINOLOGICAL",
            "linked_to":       [l for l in (arch.get("linked_to") or []) if l != art_id],
            "suggested_tags":  arch.get("suggested_tags") or [],
            "has_four_d":      bool(struct.get("has_four_d") or gen.get("four_d_matrix")),
            "stress_stable":   struct.get("stress_stable"),
            "stability_score": float(sim_d.get("stability_score") or 0.0),
        })
    return items

def load_embeddings():
    if not SEM_INDEX_PATH.exists():
        logger.error(f"Нет {SEM_INDEX_PATH}")
        return {}
    from sentence_transformers import SentenceTransformer
    logger.info("  Загружаем sentence-transformers...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    entries = []
    with open(SEM_INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: entries.append(json.loads(line))
                except: pass
    logger.info(f"  Encoding {len(entries)} текстов...")
    texts = [e["invariant"] for e in entries]
    vecs  = model.encode(texts, show_progress_bar=False)
    result = {}
    for e, vec in zip(entries, vecs):
        eid = e.get("id")
        if eid:
            result[eid] = {"vec": vec, "domain": norm_domain(e.get("domain","general"))}
    logger.info(f"  Эмбеддингов: {len(result)}")
    return result, model

def build_graph(artifacts, embeddings, st_model, threshold):
    art_by_id   = {a["id"]: a for a in artifacts}
    valid_ids   = [a["id"] for a in artifacts if a["id"] in embeddings]
    logger.info(f"  Узлов с эмбеддингами: {len(valid_ids)}")

    # Матрица сходств
    vecs  = np.array([embeddings[nid]["vec"] for nid in valid_ids])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    sim_m = (vecs / norms) @ (vecs / norms).T

    # Кэш domain distances
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

    N = len(valid_ids)
    links = []
    for i in range(N):
        for j in range(i + 1, N):
            sim = float(sim_m[i, j])
            if sim < threshold:
                continue
            id_i = valid_ids[i]
            id_j = valid_ids[j]
            a_i  = art_by_id[id_i]
            a_j  = art_by_id[id_j]
            dist = get_dist(a_i["domain"], a_j["domain"])
            spec = (a_i["specificity"] + a_j["specificity"]) / 2
            links.append({
                "source":          id_i,
                "target":          id_j,
                "similarity":      round(sim, 3),
                "domain_distance": dist,
                "specificity":     round(spec, 3),
                "four_d_resonance": 0.0,
                "weight":          round(sim * (1 + dist) * spec, 3),
            })

    logger.info(f"  Рёбер: {len(links)} (порог={threshold})")

    nodes_out = []
    for a in artifacts:
        nodes_out.append({
            "id":              a["id"],
            "domain":          a["domain"],
            "b_sync":          a["b_sync"],
            "stability":       a["stability"],
            "artifact_type":   a["artifact_type"],
            "specificity":     a["specificity"],
            "survival":        a["survival"],
            "novelty":         a["novelty"],
            "novelty_score":   a["novelty_score"],
            "math_verification": a["math_verification"],
            "linked_to":       a["linked_to"],
            "suggested_tags":  a["suggested_tags"],
            "has_four_d":      a["has_four_d"],
            "stress_stable":   a["stress_stable"],
            "stability_score": a["stability_score"],
        })

    return {
        "directed": False, "multigraph": False, "graph": {},
        "nodes": nodes_out,
        "links": links,
        "_restored_at": datetime.now(timezone.utc).isoformat(),
        "_restore_threshold": threshold,
        "_node_count": len(nodes_out),
        "_edge_count": len(links),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    print("\n🔧 HX-AM Graph Restore")
    print(f"   порог: {args.threshold}")

    print("\n📦 Читаем артефакты...")
    artifacts = load_artifacts()
    if not artifacts:
        print("❌ Нет артефактов")
        sys.exit(1)
    print(f"   {len(artifacts)} артефактов")
    by_type = Counter(a["artifact_type"] for a in artifacts)
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"   {t:35s} {c}")

    print("\n🧠 Загружаем эмбеддинги...")
    result = load_embeddings()
    embeddings, st_model = result

    print("\n🕸️  Строим граф...")
    graph_data = build_graph(artifacts, embeddings, st_model, args.threshold)
    n  = graph_data["_node_count"]
    e  = graph_data["_edge_count"]
    print(f"\n   {n} узлов, {e} рёбер  (было 105 узлов, 5160 рёбер)")

    if args.dry_run:
        print("(dry-run: файл не записан)")
        return

    if GRAPH_PATH.exists() and not args.no_backup:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = GRAPH_PATH.parent / f"invariant_graph.{ts}.bak.json"
        shutil.copy2(GRAPH_PATH, bak)
        print(f"💾 Backup: {bak}")

    GRAPH_PATH.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2))
    print(f"✅ Записано: {GRAPH_PATH}")
    print("   Перезапустите сервер: python hxam_v_4_server.py")

if __name__ == "__main__":
    main()
