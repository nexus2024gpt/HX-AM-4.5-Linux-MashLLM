#!/usr/bin/env python3
# tools/dedupe_indexes.py — HX-AM
"""
Убирает дубли id в semantic_index.jsonl / four_d_index.jsonl.

ПРОБЛЕМА КОТОРУЮ РЕШАЕТ:
  Индексы дописываются построчно, и при повторной обработке артефакта
  (REF-обновление, миграция, ручная пересборка) появляется вторая запись
  с тем же id. Дубли бывают неидентичными: например, у f7d9f1aa4394 одна
  запись domain='ecology' b_sync=0.81, вторая — domain='экосистема'
  (ненормализованный русский) b_sync=0.0, реликт до нормализации доменов.
  SemanticSpace строит _id_to_idx как {id: i}, поэтому при дублях
  побеждает последняя запись — то есть как раз мусорная.

СТРАТЕГИЯ ВЫБОРА:
  Из дублей оставляем запись, согласованную с самим артефактом
  (domain и b_sync из artifacts/<id>.json). Если ни одна не совпала —
  первую по порядку.

CLI:
  python tools/dedupe_indexes.py --dry-run
  python tools/dedupe_indexes.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BACKUP_DIR    = PROJECT_ROOT / "backups"
INDEXES = ["semantic_index.jsonl", "four_d_index.jsonl"]


def artifact_facts(aid: str) -> dict:
    """domain/b_sync из самого артефакта — эталон для выбора дубля."""
    for name in (f"{aid}.json", f"{aid}.hyx-portal.json"):
        p = ARTIFACTS_DIR / name
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
        data = d.get("data") or {}
        gen  = data.get("gen") or {}
        return {
            "domain": gen.get("domain") or data.get("domain") or d.get("domain"),
            "b_sync": gen.get("b_sync"),
        }
    return {}


def score(row: dict, facts: dict) -> int:
    """Насколько запись индекса согласована с артефактом."""
    if not facts:
        return 0
    s = 0
    if facts.get("domain") and str(row.get("domain")) == str(facts["domain"]):
        s += 2
    b = facts.get("b_sync")
    if b is not None:
        try:
            if abs(float(row.get("b_sync", -1)) - float(b)) < 1e-6:
                s += 1
        except (TypeError, ValueError):
            pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changes = {}
    for idx_name in INDEXES:
        path = ARTIFACTS_DIR / idx_name
        if not path.exists():
            continue
        rows = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

        seen: dict[str, int] = {}
        order: list[dict] = []
        dupes = 0
        for r in rows:
            rid = r.get("id")
            if rid in seen:
                dupes += 1
                pos    = seen[rid]
                facts  = artifact_facts(rid)
                if score(r, facts) > score(order[pos], facts):
                    print(f"  {idx_name}: {rid} — оставляю более позднюю "
                          f"(domain={r.get('domain')!r} b_sync={r.get('b_sync')})")
                    order[pos] = r
                else:
                    print(f"  {idx_name}: {rid} — оставляю первую "
                          f"(domain={order[pos].get('domain')!r} "
                          f"b_sync={order[pos].get('b_sync')})")
            else:
                seen[rid] = len(order)
                order.append(r)

        if dupes:
            changes[idx_name] = (path, order, len(rows), len(order), dupes)
        print(f"  {idx_name}: {len(rows)} строк, дублей {dupes}")

    if not changes:
        print("\nДублей нет.")
        return
    if args.dry_run:
        print("\n(dry-run: ничего не изменено)")
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir  = BACKUP_DIR / f"dedupe_{stamp}"
    bdir.mkdir()
    for idx_name, (path, order, before, after, dupes) in changes.items():
        shutil.copy2(path, bdir / path.name)
        with open(path, "w", encoding="utf-8") as f:
            for r in order:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✓ {idx_name}: {before} → {after} строк (-{dupes})")
    print(f"✓ бэкап: {bdir}")


if __name__ == "__main__":
    main()
