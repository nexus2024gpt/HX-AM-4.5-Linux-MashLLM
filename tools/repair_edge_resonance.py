#!/usr/bin/env python3
# tools/repair_edge_resonance.py — HX-AM
"""
Пересчитывает four_d_resonance и weight для рёбер существующего графа.

ПРОБЛЕМА КОТОРУЮ РЕШАЕТ:
  tools/rebuild_graph_knn.py до правки захардкоживал "four_d_resonance": 0.0,
  поэтому все рёбра, созданные пересборкой, потеряли 4D-буст веса
  (weight = base × (1 + resonance × 0.2), см. InvariantGraph.add_edge).
  Рёбра от живого пайплайна резонанс имели → граф стал внутренне
  несопоставим: часть весов с бустом, часть без.

ЧТО ДЕЛАЕТ:
  Только пересчитывает four_d_resonance и weight на рёбрах. Топологию
  (набор узлов и рёбер) НЕ меняет — в отличие от полной пересборки.

  Для рёбер, у которых резонанс уже ненулевой, по умолчанию ничего не
  трогает (--all чтобы пересчитать все).

CLI:
  python tools/repair_edge_resonance.py --dry-run   # показать что изменится
  python tools/repair_edge_resonance.py             # применить (с бэкапом)
  python tools/repair_edge_resonance.py --all       # пересчитать все рёбра
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT      = Path(__file__).parent.parent
ARTIFACTS_DIR     = PROJECT_ROOT / "artifacts"
GRAPH_PATH        = ARTIFACTS_DIR / "invariant_graph.json"
FOUR_D_INDEX_PATH = ARTIFACTS_DIR / "four_d_index.jsonl"
BACKUP_DIR        = PROJECT_ROOT / "backups"

sys.path.insert(0, str(PROJECT_ROOT))

# Должен совпадать с InvariantGraph.add_edge() в invariant_engine.py
RESONANCE_BOOST = 0.2


def load_four_d_vectors() -> dict:
    if not FOUR_D_INDEX_PATH.exists():
        print(f"✗ не найден {FOUR_D_INDEX_PATH}")
        sys.exit(1)
    out = {}
    for line in open(FOUR_D_INDEX_PATH, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        vid, vec = o.get("id"), o.get("vector")
        if vid and vec:
            out[vid] = np.asarray(vec, dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="показать без записи")
    ap.add_argument("--all", action="store_true",
                    help="пересчитать все рёбра, а не только с нулевым резонансом")
    args = ap.parse_args()

    try:
        from schemas.four_d_matrix import compute_4d_resonance
    except Exception as e:
        print(f"✗ не импортируется schemas.four_d_matrix: {e}")
        sys.exit(1)

    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    edges = graph.get("edges") or graph.get("links") or []
    vecs  = load_four_d_vectors()
    print(f"  рёбер: {len(edges)}, 4D-векторов: {len(vecs)}")

    updated = skipped_novec = unchanged = 0
    deltas = []

    for e in edges:
        old_res = e.get("four_d_resonance", 0.0) or 0.0
        if not args.all and old_res > 0:
            unchanged += 1
            continue
        s, t = e.get("source"), e.get("target")
        va, vb = vecs.get(s), vecs.get(t)
        if va is None or vb is None:
            skipped_novec += 1
            continue
        try:
            res = float(compute_4d_resonance(va, vb))
        except Exception as ex:
            print(f"  ! {s}~{t}: {ex}")
            skipped_novec += 1
            continue

        # weight пересчитывается от базы, а не от текущего значения,
        # иначе повторный запуск накрутит буст поверх буста
        sim  = e.get("similarity", 0.0)
        dist = e.get("domain_distance", 0.0)
        spec = e.get("specificity", 0.5)
        base = sim * (1 + dist) * spec
        new_weight = round(base * (1.0 + res * RESONANCE_BOOST), 4)

        old_weight = e.get("weight", base)
        if old_weight:
            deltas.append((new_weight - old_weight) / old_weight)
        e["four_d_resonance"] = res
        e["weight"] = new_weight
        updated += 1

    print(f"\n  обновлено рёбер:            {updated}")
    print(f"  пропущено (нет 4D-вектора): {skipped_novec}")
    print(f"  не тронуто (резонанс > 0):  {unchanged}")
    if deltas:
        d = sorted(deltas)
        print(f"  изменение веса: медиана {100*d[len(d)//2]:+.1f}%  "
              f"мин {100*min(d):+.1f}%  макс {100*max(d):+.1f}%")

    if args.dry_run:
        print("\n(dry-run: файл не записан)")
        return
    if not updated:
        print("\nНечего записывать.")
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"invariant_graph_{stamp}.json"
    shutil.copy2(GRAPH_PATH, backup)
    GRAPH_PATH.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ записано: {GRAPH_PATH}")
    print(f"✓ бэкап:    {backup}")


if __name__ == "__main__":
    main()
