#!/usr/bin/env python3
# tools/purge_artifacts.py — HX-AM
"""
Согласованное удаление артефактов из всех четырёх хранилищ сразу:
  artifacts/<id>.json  →  trash/
  artifacts/invariant_graph.json   (узел + инцидентные рёбра)
  artifacts/semantic_index.jsonl
  artifacts/four_d_index.jsonl

ЗАЧЕМ ОТДЕЛЬНЫЙ ИНСТРУМЕНТ:
  Штатный DELETE /artifact/{id} в hxam_v_4_server.py чистит файл, граф и
  semantic_index, но НЕ трогает four_d_index.jsonl — записи там копятся.
  Плюс он не умеет чистить «сирот»: записи в графе/индексах, у которых
  файла артефакта уже нет (удалён мимо API).

РЕЖИМЫ:
  --id <id> [<id>...]   удалить конкретные артефакты (файл → trash/)
  --orphans             найти и вычистить все записи без файла артефакта

CLI:
  python tools/purge_artifacts.py --orphans --dry-run
  python tools/purge_artifacts.py --orphans
  python tools/purge_artifacts.py --id 3713bd6d555d --dry-run
  python tools/purge_artifacts.py --id 3713bd6d555d
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
TRASH_DIR         = PROJECT_ROOT / "trash"
BACKUP_DIR        = PROJECT_ROOT / "backups"
GRAPH_PATH        = ARTIFACTS_DIR / "invariant_graph.json"
SEM_INDEX_PATH    = ARTIFACTS_DIR / "semantic_index.jsonl"
FOUR_D_INDEX_PATH = ARTIFACTS_DIR / "four_d_index.jsonl"


def artifact_ids_on_disk() -> set:
    """id всех артефактов, у которых есть файл (по внутреннему id, а не имени)."""
    ids = set()
    for p in ARTIFACTS_DIR.glob("*.json"):
        if p.name == "invariant_graph.json":
            continue
        stem = p.stem.replace(".hyx-portal", "")
        ids.add(stem)
        try:
            internal = json.loads(p.read_text(encoding="utf-8")).get("id")
            if internal:
                ids.add(internal)   # у части артефактов id ≠ имени файла
        except Exception:
            pass
    return ids


def read_jsonl(path: Path) -> list:
    out = []
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def write_jsonl(path: Path, rows: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", nargs="+", default=[], help="id артефактов к удалению")
    ap.add_argument("--orphans", action="store_true",
                    help="вычистить записи, у которых нет файла артефакта")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.id and not args.orphans:
        ap.error("нужен --id или --orphans")

    graph   = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes   = graph.get("nodes", [])
    edges   = graph.get("edges") or graph.get("links") or []
    sem     = read_jsonl(SEM_INDEX_PATH)
    four_d  = read_jsonl(FOUR_D_INDEX_PATH)
    on_disk = artifact_ids_on_disk()

    targets = set(args.id)
    if args.orphans:
        in_stores = ({n["id"] for n in nodes}
                     | {r.get("id") for r in sem}
                     | {r.get("id") for r in four_d})
        orphans = {i for i in in_stores if i and i not in on_disk}
        print(f"  сирот найдено: {len(orphans)}")
        targets |= orphans

    if not targets:
        print("  нечего удалять")
        return

    # ── файлы → trash ────────────────────────────────────────────────
    to_trash = []
    for aid in sorted(targets):
        for name in (f"{aid}.json", f"{aid}.hyx-portal.json"):
            p = ARTIFACTS_DIR / name
            if p.exists():
                to_trash.append(p)

    # ── что вырежется из хранилищ ────────────────────────────────────
    node_hits = [n for n in nodes if n["id"] in targets]
    edge_hits = [e for e in edges
                 if e.get("source") in targets or e.get("target") in targets]
    sem_hits  = [r for r in sem if r.get("id") in targets]
    fd_hits   = [r for r in four_d if r.get("id") in targets]

    print(f"\n  целей:                  {len(targets)}")
    print(f"  файлов → trash/:        {len(to_trash)}")
    print(f"  узлов графа:            {len(node_hits)}")
    print(f"  рёбер (инцидентных):    {len(edge_hits)}")
    print(f"  записей semantic_index: {len(sem_hits)}")
    print(f"  записей four_d_index:   {len(fd_hits)}")

    if args.dry_run:
        for aid in sorted(targets)[:20]:
            print(f"      {aid}")
        if len(targets) > 20:
            print(f"      … ещё {len(targets)-20}")
        print("\n(dry-run: ничего не изменено)")
        return

    # ── бэкап ────────────────────────────────────────────────────────
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir  = BACKUP_DIR / f"purge_{stamp}"
    bdir.mkdir()
    for src in (GRAPH_PATH, SEM_INDEX_PATH, FOUR_D_INDEX_PATH):
        if src.exists():
            shutil.copy2(src, bdir / src.name)

    # ── применяем ────────────────────────────────────────────────────
    TRASH_DIR.mkdir(exist_ok=True)
    for p in to_trash:
        shutil.move(str(p), str(TRASH_DIR / p.name))

    graph["nodes"] = [n for n in nodes if n["id"] not in targets]
    kept_edges = [e for e in edges
                  if e.get("source") not in targets and e.get("target") not in targets]
    if "edges" in graph:
        graph["edges"] = kept_edges
    else:
        graph["links"] = kept_edges
    GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    write_jsonl(SEM_INDEX_PATH, [r for r in sem if r.get("id") not in targets])
    write_jsonl(FOUR_D_INDEX_PATH, [r for r in four_d if r.get("id") not in targets])

    print(f"\n✓ применено")
    print(f"✓ граф:   {len(graph['nodes'])} узлов, {len(kept_edges)} рёбер")
    print(f"✓ бэкап:  {bdir}")


if __name__ == "__main__":
    main()
