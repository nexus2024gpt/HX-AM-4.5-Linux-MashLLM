# tools/clean_artifacts.py — HX-AM v4.2
"""
Очищает артефакты от артефактов LLM-форматирования.

ПРОБЛЕМА: LLM при генерации/миграции JSON добавляет пробел
после каждого ключа ("key " вместо "key"). Файл парсируется Python,
но server-код: data.get("gen") → None, data.get("four_d_matrix") → None.

ЧТО ЧИСТИМ:
  1. Trailing/leading пробелы в ключах JSON ("key " → "key")
  2. Trailing/leading пробелы в строковых значениях ("VALID " → "VALID")
  3. Двойные пробелы внутри строк ("осцилл яторов" → "осцилляторов")

ЗАПУСК:
  python tools/clean_artifacts.py --dry-run      # показать что грязное
  python tools/clean_artifacts.py                # очистить всё
  python tools/clean_artifacts.py --id 05377791a193
"""

import json
import re
import sys
import argparse
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")


def deep_clean(obj):
    if isinstance(obj, dict):
        return {k.strip(): deep_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean(i) for i in obj]
    elif isinstance(obj, str):
        cleaned = obj.strip()
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        return cleaned
    return obj


def count_bad_keys(obj) -> int:
    count = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k != k.strip():
                count += 1
            count += count_bad_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            count += count_bad_keys(item)
    return count


def needs_cleaning(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        bad = count_bad_keys(data)
        return bad > 0, bad
    except Exception:
        return False, 0


def clean_file(path: Path, dry_run: bool = False) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        bad_before = count_bad_keys(data)
        if bad_before == 0:
            return False
        cleaned = deep_clean(data)
        bad_after = count_bad_keys(cleaned)
        if not dry_run:
            path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2))
        prefix = "[dry] " if dry_run else ""
        print(f"  {prefix}OK {path.name}: {bad_before} -> {bad_after} bad keys")
        return True
    except Exception as e:
        print(f"  ERR {path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="HX-AM Artifact Cleaner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--id", type=str, default="")
    args = parser.parse_args()

    print("\nHX-AM Artifact Cleaner v4.2")

    if args.id:
        for path in [ARTIFACTS_DIR / f"{args.id}.json",
                     ARTIFACTS_DIR / f"{args.id}.hyx-portal.json"]:
            if path.exists():
                dirty, n = needs_cleaning(path)
                if not dirty:
                    print(f"OK {args.id}: clean")
                else:
                    print(f"DIRTY {args.id}: {n} bad keys")
                    clean_file(path, dry_run=args.dry_run)
                return
        print(f"NOT FOUND: {args.id}")
        sys.exit(1)

    files = list(ARTIFACTS_DIR.glob("*.json")) + list(ARTIFACTS_DIR.glob("*.hyx-portal.json"))
    files = [f for f in files if f.stem != "invariant_graph"]
    dirty_files = [(f, n) for f in sorted(files) for d, n in [needs_cleaning(f)] if d]

    print(f"Total: {len(files)}  Dirty: {len(dirty_files)}")
    if not dirty_files:
        print("All clean")
        return

    ok = 0
    for f, n in dirty_files:
        if clean_file(f, dry_run=args.dry_run):
            ok += 1

    if not args.dry_run:
        print(f"\nCleaned: {ok} files")


if __name__ == "__main__":
    main()
