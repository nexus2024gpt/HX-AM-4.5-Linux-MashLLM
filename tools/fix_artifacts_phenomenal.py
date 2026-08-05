# tools/fix_artifacts_phenomenal.py — HX-AM v4.2
"""
Ретроактивная коррекция PHENOMENAL-инфляции в существующих артефактах.

Три критерия снижения PHENOMENAL -> NOVEL:
  1. simulation.stability_score < 0.5  (математически нестабильна)
  2. 4D matrix == шаблон из промпта   (14/14 параметров совпадают → LLM скопировал template)
  3. cross_domain_links не упомянуты в hypothesis/mechanism тексте

CLI:
  python tools/fix_artifacts_phenomenal.py --dry-run    # показать без изменений
  python tools/fix_artifacts_phenomenal.py              # применить
  python tools/fix_artifacts_phenomenal.py --id b98249ff87b1
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# sys.path[0] — это tools/, поэтому корень проекта нужно добавить явно,
# иначе `from response_normalizer import DOMAIN_MAP` молча не резолвится
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fix_phenomenal")

ARTIFACTS_DIR = Path("artifacts")

# Точная копия шаблона из старого generator_prompt.txt OUTPUT FORMAT
PROMPT_TEMPLATE_4D = {
    "structure": {"C": 0.62, "k": 9.0, "D": 2.15},
    "influence": {"h": 0.95, "T": 1.05, "eta": 0.18},
    "dynamics": {"omega_i": 0.25, "K": 0.58, "K_c": 0.48, "p": 0.72, "model": "kuramoto"},
    "time": {"tau": 0.55, "H": 0.79, "freq": 1.15},
}
STABILITY_THRESHOLD = 0.5


def _is_template_4d(four_d: dict) -> bool:
    if not four_d or not isinstance(four_d, dict):
        return False
    matches = total = 0
    for layer, params in PROMPT_TEMPLATE_4D.items():
        art_layer = four_d.get(layer, {})
        if not isinstance(art_layer, dict):
            continue
        for k, v in params.items():
            total += 1
            art_v = art_layer.get(k)
            if art_v is None:
                continue
            try:
                if abs(float(art_v) - float(v)) < 0.001:
                    matches += 1
            except (TypeError, ValueError):
                pass
    return total > 0 and matches == total


def _domain_stems(domain: str) -> list:
    """
    Основы для поиска домена в тексте: сам слаг + русские варианты.

    Домены хранятся английскими слагами ('materials_science'), а гипотезы
    написаны по-русски, поэтому буквальный поиск слага не находит ничего
    никогда. Проверено на корпусе: слаг встречается в 0 из 78 PHENOMENAL,
    русский корень — в 38 из 78. Без этой нормализации критерий понижает
    100% артефактов и не несёт никакой информации.
    """
    d = (domain or "").strip().lower()
    stems = {d, d.replace("_", " ")}
    for ru, en in _RU_BY_EN.get(d, []):
        stems.add(ru)
    return [s for s in stems if len(s) >= 4]


# Домены корпуса, которых нет в DOMAIN_MAP response_normalizer'а.
# Без них критерий ложно срабатывает: materials_science встречается
# в 18 связях PHENOMENAL-артефактов, robotics — в 17.
_RU_SUPPLEMENT = {
    "materials_science": ["материал"],
    "robotics":          ["робот"],
    "engineering":       ["инженер"],
    "oceanography":      ["океан"],
    "computer_science":  ["вычислит", "информатик"],
    "cognitive_science": ["когнитив"],
    "immunology":        ["иммун"],
    "sociolinguistics":  ["социолингв"],
    "political_science": ["политол"],
    "systems_theory":    ["системолог"],
}


def _build_ru_index() -> dict:
    """English-слаг → русские основы, из общего словаря проекта + дополнения."""
    out = {}
    try:
        from response_normalizer import DOMAIN_MAP
    except Exception:
        DOMAIN_MAP = {}
    for ru, en in DOMAIN_MAP.items():
        if not _is_cyrillic(ru):
            continue
        # усечённая основа, чтобы ловить словоформы:
        # 'биология' → 'биолог', 'физика' → 'физи'
        stem = ru[:-2] if len(ru) > 6 else ru[:-1] if len(ru) > 4 else ru
        out.setdefault(en, []).append((stem, en))
    for en, stems in _RU_SUPPLEMENT.items():
        for s in stems:
            out.setdefault(en, []).append((s, en))
    return out


def _is_cyrillic(s: str) -> bool:
    return any("а" <= ch <= "я" or ch == "ё" for ch in s.lower())


_RU_BY_EN = _build_ru_index()


def _cross_domain_in_text(cross_domain_links: list, hypothesis: str, mechanism: str) -> bool:
    if not cross_domain_links:
        return False
    full_text = (hypothesis + " " + mechanism).lower()
    for link in cross_domain_links:
        # Поддерживаем форматы: "biology->economics", "biology→economics"
        for sep in ["->", "→", " → ", " -> "]:
            if sep in link:
                parts = link.split(sep, 1)
                d1, d2 = parts[0].strip().lower(), parts[1].strip().lower()
                for dom in (d1, d2):
                    if any(stem in full_text for stem in _domain_stems(dom)):
                        return True
    return False


def process_artifact(path: Path, dry_run: bool) -> dict:
    try:
        art = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"file": path.name, "error": str(e), "action": "error"}

    art_id = art.get("id", path.stem)
    arch = art.get("archivist", {})

    if not arch or arch.get("novelty") != "PHENOMENAL":
        return {"file": path.name, "id": art_id,
                "novelty": (arch or {}).get("novelty", "none"), "action": "skip"}

    data = art.get("data", {})
    gen = data.get("gen", {})
    hypothesis = gen.get("hypothesis", "")
    mechanism = gen.get("mechanism", "")
    four_d = gen.get("four_d_matrix")
    cross_links = arch.get("cross_domain_links", [])
    sim = art.get("simulation") or {}
    stability_score = sim.get("stability_score", 1.0)

    reasons = []

    if stability_score < STABILITY_THRESHOLD:
        reasons.append(f"stability_score={stability_score:.3f} < {STABILITY_THRESHOLD}")

    if _is_template_4d(four_d):
        reasons.append("4D matrix = prompt template (14/14 params identical)")

    if cross_links and not _cross_domain_in_text(cross_links, hypothesis, mechanism):
        reasons.append(f"cross_domain {cross_links} not in hypothesis/mechanism text")

    if not reasons:
        return {"file": path.name, "id": art_id, "novelty": "PHENOMENAL",
                "action": "keep", "score": stability_score}

    if not dry_run:
        arch["novelty"] = "NOVEL"
        arch["_downgraded_from"] = "PHENOMENAL"
        arch["_downgrade_reason"] = "; ".join(reasons)
        arch["_downgrade_by"] = "tools/fix_artifacts_phenomenal.py"
        art["archivist"] = arch
        path.write_text(json.dumps(art, ensure_ascii=False, indent=2))

    return {
        "file": path.name, "id": art_id,
        "action": "downgrade_dry" if dry_run else "downgraded",
        "reasons": reasons, "stability_score": stability_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Fix PHENOMENAL inflation in HX-AM artifacts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--id", type=str, default="")
    args = parser.parse_args()

    print("\n== HX-AM PHENOMENAL Inflation Fix ==")
    print(f"Criteria: stability<{STABILITY_THRESHOLD} OR 4D=template OR cross-domain not in text")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    if args.id:
        files = [ARTIFACTS_DIR / f"{args.id}.json"]
    else:
        files = sorted([f for f in ARTIFACTS_DIR.glob("*.json")
                        if "hyx-portal" not in f.name and f.stem != "invariant_graph"])

    kept = downgraded = errors = 0

    for f in files:
        if not f.exists():
            print(f"  NOT FOUND: {f.name}")
            continue
        r = process_artifact(f, dry_run=args.dry_run)
        action = r.get("action", "")

        if action in ("downgrade_dry", "downgraded"):
            downgraded += 1
            tag = "[DRY]" if args.dry_run else "[FIX]"
            print(f"  {tag} {r['id']}")
            for reason in r.get("reasons", []):
                print(f"       — {reason}")
        elif action == "keep":
            kept += 1
            print(f"  [OK]  {r['id']} PHENOMENAL kept (score={r.get('score', '?')})")
        elif action == "error":
            errors += 1
            print(f"  [ERR] {r['file']}: {r.get('error')}")

    print(f"\nResult: {downgraded} {'would be ' if args.dry_run else ''}downgraded, "
          f"{kept} kept PHENOMENAL, {errors} errors")
    if args.dry_run and downgraded > 0:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
