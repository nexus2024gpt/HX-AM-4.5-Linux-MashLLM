#!/usr/bin/env python3
# tools/add_missing_nodes.py — HX-AM
"""
Добавляет в invariant_graph.json артефакты, которые есть на диске и в
semantic_index, но отсутствуют в графе.

ПОЧЕМУ НЕ ПОЛНАЯ ПЕРЕСБОРКА:
  tools/rebuild_graph_knn.py пересобрал бы граф целиком по k-NN, заменив
  ВСЕ рёбра. Живой пайплайн (process_with_invariants в invariant_engine.py)
  строит рёбра иначе — по порогу similarity >= 0.65 от space.nearest().
  Это разная семантика графа, и менять её ради пары пропущенных узлов
  неправильно. Здесь узлы добавляются ровно так, как это сделал бы
  пайплайн при штатной обработке.

  Атрибуты узла (specificity, stability, survival) берутся из самого
  артефакта — это решения, принятые пайплайном в момент создания, их
  не нужно пересчитывать задним числом.

CLI:
  python tools/add_missing_nodes.py --dry-run
  python tools/add_missing_nodes.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BACKUP_DIR    = PROJECT_ROOT / "backups"
GRAPH_PATH    = ARTIFACTS_DIR / "invariant_graph.json"

sys.path.insert(0, str(PROJECT_ROOT))

SIM_THRESHOLD = 0.65   # как в process_with_invariants()
RESONANCE_BOOST = 0.2  # как в InvariantGraph.add_edge()


def load_artifact(aid: str):
    for name in (f"{aid}.json", f"{aid}.hyx-portal.json"):
        p = ARTIFACTS_DIR / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from scipy.spatial.distance import cosine
    from invariant_engine import SemanticSpace, _get_domain_vec, _normalize_domain

    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges") if "edges" in graph else graph.get("links", [])
    in_graph = {n["id"] for n in nodes}

    space = SemanticSpace()
    indexed = {m["id"] for m in space.meta}

    on_disk = set()
    for p in ARTIFACTS_DIR.glob("*.json"):
        if p.name == "invariant_graph.json":
            continue
        d = load_artifact(p.stem.replace(".hyx-portal", ""))
        if d:
            on_disk.add(d.get("id") or p.stem.replace(".hyx-portal", ""))

    missing = sorted((on_disk & indexed) - in_graph)
    print(f"  узлов в графе: {len(in_graph)}, артефактов: {len(on_disk)}")
    print(f"  к добавлению:  {len(missing)} {missing}")
    if not missing:
        return

    try:
        from schemas.four_d_matrix import compute_4d_resonance
        have_res = True
    except Exception as e:
        print(f"  ! резонанс недоступен ({e}) — рёбра получат 0.0")
        have_res = False

    new_nodes, new_edges = [], []
    for aid in missing:
        art = load_artifact(aid)
        data   = art.get("data") or {}
        gen    = data.get("gen") or {}
        struct = data.get("structural") or {}
        arch   = art.get("archivist") or {}
        sim_d  = art.get("simulation") or {}

        domain    = _normalize_domain(gen.get("domain") or data.get("domain") or "general")
        invariant = gen.get("hypothesis") or ""
        spec      = float(struct.get("specificity") or 0.5)

        node = {
            "id":              aid,
            "domain":          domain,
            "b_sync":          float(gen.get("b_sync") or 0.5),
            "stability":       struct.get("stability") or "unknown",
            "specificity":     spec,
            "survival":        (struct.get("translation") or {}).get("survival", "UNKNOWN"),
            "artifact_type":   struct.get("artifact_type") or "unknown",
            "novelty":         (arch.get("novelty") or "").split(":")[0],
            "novelty_score":   float(arch.get("novelty_score") or 0.5),
            "stability_score": float(sim_d.get("stability_score") or 0.0),
            "has_four_d":      bool(gen.get("four_d_matrix")),
            "stress_stable":   struct.get("stress_stable"),
        }
        new_nodes.append(node)

        similar = space.nearest(invariant, top_k=5, threshold=SIM_THRESHOLD)
        dom_vec = _get_domain_vec(domain)
        four_d_vec = space.four_d_vec_by_id(aid)
        node_by_id = {n["id"]: n for n in nodes}

        added = 0
        for s in similar:
            nid = s["id"]
            if nid == aid or nid not in in_graph:
                continue
            try:
                ndom = _normalize_domain(s.get("domain", "general"))
                dist = round(float(cosine(dom_vec, _get_domain_vec(ndom))), 3)
            except Exception:
                dist = 0.0
            nspec = float(node_by_id.get(nid, {}).get("specificity", 0.5))
            edge_spec = round((spec + nspec) / 2, 3)

            res = 0.0
            if have_res and four_d_vec is not None:
                nvec = space.four_d_vec_by_id(nid)
                if nvec is not None:
                    try:
                        res = float(compute_4d_resonance(four_d_vec, nvec))
                    except Exception:
                        res = 0.0

            base = s["similarity"] * (1 + dist) * edge_spec
            new_edges.append({
                "similarity":       s["similarity"],
                "domain_distance":  dist,
                "specificity":      edge_spec,
                "four_d_resonance": res,
                "weight":           round(base * (1 + res * RESONANCE_BOOST), 4),
                "source":           aid,
                "target":           nid,
            })
            added += 1
        print(f"    {aid} [{domain}] → рёбер: {added}")

    print(f"\n  новых узлов: {len(new_nodes)}, новых рёбер: {len(new_edges)}")
    if args.dry_run:
        print("\n(dry-run: ничего не изменено)")
        return
    if not new_nodes:
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"invariant_graph_{stamp}.json"
    shutil.copy2(GRAPH_PATH, backup)

    nodes.extend(new_nodes)
    edges.extend(new_edges)
    graph["nodes"] = nodes
    if "edges" in graph:
        graph["edges"] = edges
    else:
        graph["links"] = edges
    GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ граф: {len(nodes)} узлов, {len(edges)} рёбер")
    print(f"✓ бэкап: {backup}")


if __name__ == "__main__":
    main()
