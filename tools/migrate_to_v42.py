# tools/migrate_to_v42.py — HX-AM Artifact Migration Tool
"""
Мигрирует существующие артефакты v4.0 в формат v4.2 (добавляет four_d_matrix).

Стратегия:
  1. Читает hypothesis + mechanism из артефакта
  2. Вызывает LLM Generator с промптом 4D-извлечения
  3. Нормализует и валидирует FourDMatrix
  4. Добавляет four_d_matrix в ген-блок артефакта
  5. Добавляет вектор в four_d_index.jsonl
  6. Запускает StressTester (опционально --stress)

CLI:
  python tools/migrate_to_v42.py --dry-run          # список кандидатов
  python tools/migrate_to_v42.py                    # мигрировать всё
  python tools/migrate_to_v42.py --id 57cfa5baa346  # один артефакт
  python tools/migrate_to_v42.py --stress            # + stress-test каждого
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.four_d_matrix import FourDMatrix
from math_core import MathCore
from api_usage_tracker import tracker
from llm_client_v_4 import LLMClient

logger = logging.getLogger("HXAM.migrate")

ARTIFACTS_DIR = Path("artifacts")
FOUR_D_PROMPT = """You are an HX-AM 4D extractor. Given a hypothesis and mechanism, extract the 4D matrix.
Output ONLY valid JSON with this exact structure (all fields required):
{
  "four_d_matrix": {
    "structure": {"C": float, "k": float, "D": float},
    "influence": {"h": float, "T": float, "eta": float},
    "dynamics": {"omega_i": float, "K": float, "K_c": float, "p": float, "model": "kuramoto|percolation|ising|delay|graph_invariant|lotka_volterra"},
    "time": {"tau": float, "H": float, "freq": float}
  }
}
Rules:
- All values must be floats in range [0, 10]
- C, eta, p, H must be in [0, 1]
- K, K_c in [0, 2]
- Choose the most appropriate dynamics model
- If uncertain, use sensible defaults from typical values in the domain
- Output ONLY JSON, no explanation

Hypothesis: {hypothesis}
Mechanism: {mechanism}
Domain: {domain}
"""


def load_artifact(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}


def save_artifact(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def needs_migration(artifact: dict) -> bool:
    """Возвращает True если артефакт ещё не имеет four_d_matrix."""
    gen = artifact.get("data", {}).get("gen", {})
    return "four_d_matrix" not in gen or gen["four_d_matrix"] is None


def get_candidates() -> list:
    """Находит все артефакты без four_d_matrix."""
    if not ARTIFACTS_DIR.exists():
        return []
    candidates = []
    for f in sorted(ARTIFACTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime):
        if f.stem == "invariant_graph":
            continue
        if ".hyx-portal" in f.name:
            continue
        art = load_artifact(f)
        if art and needs_migration(art):
            data = art.get("data", {})
            gen = data.get("gen", {})
            candidates.append({
                "path": f,
                "id": art.get("id", f.stem),
                "domain": data.get("domain", "general"),
                "hypothesis_short": gen.get("hypothesis", "")[:80],
                "verdict": data.get("ver", {}).get("verdict", ""),
            })
    return candidates


def extract_4d_via_llm(
    hypothesis: str,
    mechanism: str,
    domain: str,
    llm: LLMClient,
) -> dict | None:
    """Вызывает LLM для извлечения 4D-матрицы из текста гипотезы."""
    prompt = FOUR_D_PROMPT.format(
        hypothesis=hypothesis[:400],
        mechanism=mechanism[:400],
        domain=domain,
    )
    raw, model = llm.generate(prompt)
    if not raw or raw.startswith("[Generator error]"):
        logger.warning(f"LLM failed: {raw[:80]}")
        return None

    # Парсим JSON
    import re
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        logger.warning("No JSON found in LLM response")
        return None

    try:
        parsed = json.loads(match.group(0))
        four_d_raw = parsed.get("four_d_matrix") or parsed
        matrix = FourDMatrix.from_raw(four_d_raw)
        if matrix is None:
            logger.warning("FourDMatrix.from_raw returned None")
            return None
        logger.info(f"4D extracted via {model}: model={matrix.dynamics.model}")
        return matrix.to_dict()
    except Exception as e:
        logger.error(f"4D extraction parse error: {e}")
        return None


def migrate_single(
    artifact: dict,
    path: Path,
    llm: LLMClient,
    math_core: MathCore,
    run_stress: bool = False,
) -> bool:
    """Мигрирует один артефакт. Возвращает True при успехе."""
    art_id = artifact.get("id", path.stem)
    data = artifact.get("data", {})
    gen = data.get("gen", {})

    hypothesis = gen.get("hypothesis", "")
    mechanism = gen.get("mechanism", "")
    domain = data.get("domain", "general")

    if not hypothesis:
        logger.warning(f"  {art_id}: no hypothesis — skipping")
        return False

    print(f"  [{art_id}] domain={domain} ...", end=" ", flush=True)

    # Извлекаем 4D
    four_d = extract_4d_via_llm(hypothesis, mechanism, domain, llm)
    if four_d is None:
        print("❌ LLM failed")
        return False

    # Обновляем артефакт
    artifact["data"]["gen"]["four_d_matrix"] = four_d
    artifact.setdefault("migration", {})["v42_migrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    save_artifact(path, artifact)

    # Добавляем в 4D-индекс
    stability_score = 0.5  # дефолт до stress-test
    math_core.index_artifact(
        artifact_id=art_id,
        four_d=four_d,
        domain=domain,
        stability_score=stability_score,
    )

    # Опциональный стресс-тест
    if run_stress:
        try:
            stress_result = math_core.stress_test(art_id, four_d)
            stability_score = stress_result.get("stability_score", 0.5)
            print(f"✅ 4D+stress (score={stability_score})")
        except Exception as e:
            print(f"✅ 4D (stress failed: {e})")
    else:
        print("✅ 4D")

    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="HX-AM v4.2 Artifact Migration Tool")
    parser.add_argument("--id",       type=str, default="", help="Migrate specific artifact ID")
    parser.add_argument("--dry-run",  action="store_true",  help="Show candidates without migrating")
    parser.add_argument("--stress",   action="store_true",  help="Run stress-test after migration")
    parser.add_argument("--delay",    type=float, default=1.0, help="Delay between LLM calls (seconds)")
    args = parser.parse_args()

    print("\n🔄 HX-AM v4.2 Migration Tool")
    print(f"   Artifacts dir: {ARTIFACTS_DIR.absolute()}")

    # Собираем кандидатов
    if args.id:
        path = ARTIFACTS_DIR / f"{args.id}.json"
        if not path.exists():
            print(f"❌ Artifact {args.id}.json not found")
            sys.exit(1)
        art = load_artifact(path)
        if not needs_migration(art):
            print(f"ℹ️  {args.id} already has four_d_matrix — skipping")
            sys.exit(0)
        candidates = [{"path": path, "id": args.id}]
    else:
        candidates = get_candidates()

    if not candidates:
        print("✅ All artifacts already migrated (or no artifacts found)")
        sys.exit(0)

    print(f"\nFound {len(candidates)} artifact(s) without four_d_matrix:\n")
    for c in candidates:
        art_id = c.get("id", c["path"].stem)
        domain = c.get("domain", "")
        verdict = c.get("verdict", "")
        hyp = c.get("hypothesis_short", "")
        print(f"  · {art_id}  [{domain}] {verdict}  {hyp}")

    if args.dry_run:
        print(f"\n(dry-run: {len(candidates)} artifacts would be processed)")
        sys.exit(0)

    print(f"\nStarting migration (stress={'yes' if args.stress else 'no'}, delay={args.delay}s)...\n")

    llm = LLMClient()
    math_core = MathCore()

    ok_count = 0
    fail_count = 0

    for c in candidates:
        path = c["path"]
        art = load_artifact(path)
        if not art:
            fail_count += 1
            continue

        ok = migrate_single(
            artifact=art,
            path=path,
            llm=llm,
            math_core=math_core,
            run_stress=args.stress,
        )

        if ok:
            ok_count += 1
        else:
            fail_count += 1

        if args.delay > 0 and ok:
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print(f"✅ Migrated: {ok_count}")
    print(f"❌ Failed:   {fail_count}")
    print(f"Total:       {ok_count + fail_count}")
    print(f"\n4D index: artifacts/four_d_index.jsonl")
    if args.stress:
        print(f"Stress results: sim_results/")


if __name__ == "__main__":
    main()
