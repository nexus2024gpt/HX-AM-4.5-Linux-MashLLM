#!/usr/bin/env python3
# tools/backfill_four_d.py — HX-AM
"""
Достраивает four_d_matrix артефактам, у которых её нет.

КОНЦЕПТУАЛЬНО ЭТО ЗАКОННО:
  Начиная с v4.5 матрица НЕ запрашивается у LLM, а считается
  детерминированно через FourDBuilder.build(domain, hypothesis, mechanism)
  — см. «Программное обогащение» в process_query(). Значит для старого
  артефакта её можно восстановить тем же способом, каким она была бы
  посчитана сегодня: результат воспроизводим и не является выдумкой
  модели. Тексты (hypothesis/mechanism/implication) не трогаются.

ЧТО ОБНОВЛЯЕТСЯ:
  artifacts/<id>.json          → data.gen.four_d_matrix
  artifacts/four_d_index.jsonl → новая запись с 13-мерным вектором
  artifacts/invariant_graph.json → node.has_four_d + four_d_resonance
                                    и weight на инцидентных рёбрах

CLI:
  python tools/backfill_four_d.py --dry-run
  python tools/backfill_four_d.py
  python tools/backfill_four_d.py --id d9ce616e91c3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT      = Path(__file__).parent.parent
ARTIFACTS_DIR     = PROJECT_ROOT / "artifacts"
BACKUP_DIR        = PROJECT_ROOT / "backups"
GRAPH_PATH        = ARTIFACTS_DIR / "invariant_graph.json"
FOUR_D_INDEX_PATH = ARTIFACTS_DIR / "four_d_index.jsonl"

sys.path.insert(0, str(PROJECT_ROOT))

RESONANCE_BOOST = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", nargs="+", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from four_d_builder import FourDBuilder
    from schemas.four_d_matrix import FourDMatrix, compute_4d_resonance

    # какие артефакты чинить
    targets = []
    for p in sorted(ARTIFACTS_DIR.glob("*.json")):
        if p.name == "invariant_graph.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        aid = d.get("id") or p.stem.replace(".hyx-portal", "")
        if args.id and aid not in args.id:
            continue
        gen = (d.get("data") or {}).get("gen") or {}
        if not gen.get("four_d_matrix") and gen.get("hypothesis"):
            targets.append((p, aid, d))

    print(f"  артефактов без four_d_matrix: {len(targets)}")
    if not targets:
        return

    builder = FourDBuilder()
    fd_rows = []
    for line in open(FOUR_D_INDEX_PATH, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                fd_rows.append(json.loads(line))
            except Exception:
                pass
    fd_ids = {r.get("id") for r in fd_rows}
    fd_vecs = {r["id"]: np.asarray(r["vector"], dtype=float)
               for r in fd_rows if r.get("vector")}

    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges") if "edges" in graph else graph.get("links", [])
    node_by_id = {n["id"]: n for n in nodes}

    planned = []
    for p, aid, d in targets:
        data = d.get("data") or {}
        gen  = data["gen"]
        domain = data.get("domain") or gen.get("domain") or "general"
        matrix = builder.build(domain=domain,
                               hypothesis=gen.get("hypothesis", ""),
                               mechanism=gen.get("mechanism", ""))
        if not matrix:
            print(f"    ✗ {aid}: FourDBuilder ничего не вернул")
            continue
        mobj = FourDMatrix.from_raw(matrix)
        vec  = mobj.to_vector() if mobj is not None else None
        if vec is None:
            print(f"    ✗ {aid}: матрица не векторизуется")
            continue
        model = (matrix.get("dynamics") or {}).get("model")
        print(f"    {aid} [{domain}] model={model} vec={len(vec)}D "
              f"index={'+' if aid not in fd_ids else '=' } "
              f"node={'+' if aid in node_by_id else '-'}")
        planned.append((p, aid, d, matrix, np.asarray(vec, dtype=float), domain))

    if args.dry_run:
        print("\n(dry-run: ничего не изменено)")
        return
    if not planned:
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir  = BACKUP_DIR / f"backfill4d_{stamp}"
    bdir.mkdir()
    shutil.copy2(GRAPH_PATH, bdir / GRAPH_PATH.name)
    shutil.copy2(FOUR_D_INDEX_PATH, bdir / FOUR_D_INDEX_PATH.name)

    edges_touched = 0
    for p, aid, d, matrix, vec, domain in planned:
        shutil.copy2(p, bdir / p.name)
        d["data"]["gen"]["four_d_matrix"] = matrix
        d.setdefault("backfill", {})["four_d_matrix"] = {
            "at": datetime.now().isoformat(),
            "by": "tools/backfill_four_d.py (FourDBuilder)",
            "note": "матрица восстановлена детерминированно, тексты не менялись",
        }
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

        if aid not in fd_ids:
            fd_rows.append({
                "id": aid,
                "domain": domain,
                "four_d": matrix,
                "vector": [round(float(x), 6) for x in vec],
                "stability_score": float((d.get("simulation") or {}).get("stability_score") or 0.0),
            })
            fd_vecs[aid] = vec

        node = node_by_id.get(aid)
        if node is not None:
            node["has_four_d"] = True
            for e in edges:
                if e.get("source") != aid and e.get("target") != aid:
                    continue
                other = e["target"] if e["source"] == aid else e["source"]
                ov = fd_vecs.get(other)
                if ov is None:
                    continue
                try:
                    res = float(compute_4d_resonance(vec, ov))
                except Exception:
                    continue
                base = e.get("similarity", 0.0) * (1 + e.get("domain_distance", 0.0)) \
                       * e.get("specificity", 0.5)
                e["four_d_resonance"] = res
                e["weight"] = round(base * (1 + res * RESONANCE_BOOST), 4)
                edges_touched += 1

    with open(FOUR_D_INDEX_PATH, "w", encoding="utf-8") as f:
        for r in fd_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ артефактов обновлено: {len(planned)}")
    print(f"✓ four_d_index: {len(fd_rows)} записей")
    print(f"✓ рёбер пересчитано: {edges_touched}")
    print(f"✓ бэкап: {bdir}")


if __name__ == "__main__":
    main()
