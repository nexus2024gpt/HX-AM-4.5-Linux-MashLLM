# tools/migrate_portals.py — HX-AM v4.2
"""
Миграция hyx-portal.json артефактов в формат v4.2.

Portal-артефакты имеют другую структуру от обычных:
  Обычный артефакт: {"data": {"gen": {"hypothesis": "...", ...}}}
  Portal-артефакт:  {"hypothesis": "...", "type": "hyx-portal", ...}  ← hypothesis в корне

Скрипт добавляет four_d_matrix в корень portal-артефакта.

CLI:
  python tools/migrate_portals.py --dry-run
  python tools/migrate_portals.py
  python tools/migrate_portals.py --id 4aa36cdc7910
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS_DIR = Path("artifacts")

PORTAL_EXTRACT_PROMPT = """You are an HX-AM 4D extractor. Given a hypothesis/description, extract the 4D matrix.
Output ONLY valid JSON with this exact structure (all fields required):
{
  "four_d_matrix": {
    "structure": {"C": float, "k": float, "D": float},
    "influence": {"h": float, "T": float, "eta": float},
    "dynamics": {"omega_i": float, "K": float, "K_c": float, "p": float, "model": "kuramoto|percolation|ising|delay|graph_invariant|lotka_volterra"},
    "time": {"tau": float, "H": float, "freq": float}
  }
}
Rules: all floats in range [0,10]; C,eta,p,H in [0,1]; K,K_c in [0,2]
Choose dynamics.model that best fits the mechanism.
Output ONLY JSON.

Hypothesis: {hypothesis}
Domain: {domain}
"""


def find_portals() -> list:
    """Находит все hyx-portal файлы без four_d_matrix."""
    result = []
    for f in sorted(ARTIFACTS_DIR.glob("*.hyx-portal.json"),
                    key=lambda x: x.stat().st_mtime):
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            if "four_d_matrix" not in art:
                result.append({
                    "path": f,
                    "id": art.get("id", f.stem.split(".")[0]),
                    "domain": art.get("domain", "general"),
                    "hypothesis_short": (art.get("hypothesis", "") or "")[:80],
                })
        except Exception:
            continue
    return result


def extract_4d_for_portal(hypothesis: str, domain: str, llm) -> dict | None:
    """Вызывает LLM для извлечения 4D-матрицы из hypothesis portal-артефакта."""
    prompt = PORTAL_EXTRACT_PROMPT.format(
        hypothesis=hypothesis[:500],
        domain=domain,
    )
    raw, model = llm.generate(prompt)
    if not raw or raw.startswith("[Generator error]"):
        return None

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        four_d_raw = parsed.get("four_d_matrix") or parsed

        from schemas.four_d_matrix import FourDMatrix
        matrix = FourDMatrix.from_raw(four_d_raw)
        if matrix is None:
            return None
        print(f"    model={matrix.dynamics.model} via {model.split('/')[-1]}")
        return matrix.to_dict()
    except Exception as e:
        print(f"    parse error: {e}")
        return None


def migrate_portal(path: Path, llm) -> bool:
    """Мигрирует один portal-файл."""
    try:
        art = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"    read error: {e}")
        return False

    art_id = art.get("id", path.stem.split(".")[0])
    hypothesis = art.get("hypothesis", "")
    domain = art.get("domain", "general")

    if not hypothesis:
        print(f"  [{art_id}] ⚠ нет hypothesis — пропуск")
        return False

    print(f"  [{art_id}] domain={domain} ...", end=" ", flush=True)

    four_d = extract_4d_for_portal(hypothesis, domain, llm)
    if four_d is None:
        print("❌ LLM failed")
        return False

    art["four_d_matrix"] = four_d
    art.setdefault("migration_v42", {})["portal_migrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    path.write_text(json.dumps(art, ensure_ascii=False, indent=2))
    print("✅")
    return True


def main():
    parser = argparse.ArgumentParser(description="HX-AM v4.2 Portal Migration Tool")
    parser.add_argument("--dry-run", action="store_true", help="Показать список без изменений")
    parser.add_argument("--id",      type=str, default="", help="Конкретный portal ID")
    parser.add_argument("--delay",   type=float, default=1.0, help="Пауза между LLM-вызовами")
    args = parser.parse_args()

    print("\n🌀 HX-AM v4.2 Portal Migration Tool")

    if args.id:
        # Ищем по ID (имя файла содержит ID)
        candidates = [
            f for f in ARTIFACTS_DIR.glob("*.hyx-portal.json")
            if args.id in f.stem
        ]
        if not candidates:
            print(f"❌ portal {args.id} не найден")
            sys.exit(1)
        portals = [{"path": candidates[0], "id": args.id}]
    else:
        portals = find_portals()

    if not portals:
        print("✅ Все portal-файлы уже имеют four_d_matrix")
        sys.exit(0)

    print(f"\nНайдено {len(portals)} portal(s) без four_d_matrix:\n")
    for p in portals:
        print(f"  · {p['id']}  [{p.get('domain','')}]  {p.get('hypothesis_short','')}")

    if args.dry_run:
        print(f"\n(dry-run: {len(portals)} файлов были бы обновлены)")
        sys.exit(0)

    from llm_client_v_4 import LLMClient
    llm = LLMClient()

    ok = fail = 0
    for p in portals:
        success = migrate_portal(p["path"], llm)
        if success:
            ok += 1
        else:
            fail += 1
        if args.delay > 0 and success:
            time.sleep(args.delay)

    print(f"\n✅ Обновлено: {ok}  ❌ Ошибок: {fail}")


if __name__ == "__main__":
    main()
