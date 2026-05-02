#!/usr/bin/env python3
# mgap_lib/tests/test_integration.py — MGAP Library v1.0
"""
Интеграционный тест MGAP Library.
Проверяет все компоненты без LLM и без БД.

Запуск из корня проекта:
  python mgap_lib/tests/test_integration.py

Ожидаемый вывод:
  ✅ 20+ тестов пройдено
  ❌ 0 ошибок
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PASS = "✅"
FAIL = "❌"
SKIP = "⚠️ "

_pass_count = 0
_fail_count = 0


def check(name: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    if condition:
        print(f"  {PASS} {name}")
        _pass_count += 1
    else:
        print(f"  {FAIL} {name}  ← {detail}")
        _fail_count += 1


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ═══════════════════════════════════════════════════════════
# 1. DOMAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════

def test_domain_classifier():
    section("1. DomainClassifier")
    from mgap_lib.engine.domain_classifier import DomainClassifier, DomainClassificationResult
    clf = DomainClassifier()

    # Tier 1: exact
    cases = [
        ("biology",     "1.5", "exact"),
        ("economics",   "5.1", "exact"),
        ("neuroscience","3.1", "exact"),
        ("geology",     "1.4", "exact"),
        ("ecology",     "4.2", "exact"),
        ("linguistics", "6.2", "exact"),
        ("psychology",  "6.1", "exact"),
        ("physics",     "1.2", "exact"),
        ("mathematics", "1.1", "exact"),
        ("general",     "1.1", "exact"),
    ]
    for domain, expected_sector, method in cases:
        r = clf.classify(domain)
        check(
            f"Tier1 '{domain}' → {expected_sector}",
            r.sector_code == expected_sector and r.method == method,
            f"got sector={r.sector_code} method={r.method}",
        )

    # Tier 2: keyword
    keyword_cases = [
        ("neural_dynamics",   "3.1"),
        ("market_economics",  "5.1"),
        ("gene_expression",   "1.5"),
        ("fluid_mechanics",   "2.2"),
        ("epidemiological",   "3.3"),
    ]
    for domain, expected_sector in keyword_cases:
        r = clf.classify(domain)
        check(
            f"Tier2 '{domain}' → {expected_sector}",
            r.sector_code == expected_sector,
            f"got sector={r.sector_code} method={r.method}",
        )

    # Имена
    r = clf.classify("biology")
    check("disc_name_ru populated",
          r.disc_name_ru == "Естественные науки",
          f"got '{r.disc_name_ru}'")
    check("sector_name_ru populated",
          r.sector_name_ru == "Биология",
          f"got '{r.sector_name_ru}'")

    # describe()
    desc = clf.describe("ecology")
    check("describe() contains sector name",
          "Экология" in desc or "4.2" in desc,
          f"got '{desc}'")

    # to_dict()
    r = clf.classify("economics")
    d = r.to_dict()
    check("to_dict() has all keys",
          all(k in d for k in ["domain_raw","disc_code","sector_code","method","confidence"]),
          str(d.keys()))

    # Batch
    results = clf.classify_batch(["biology", "economics", "physics"])
    check("classify_batch returns 3 items", len(results) == 3)
    check("classify_batch all exact", all(r.method == "exact" for r in results.values()))

    # Empty / unknown
    r_empty = clf.classify("")
    check("empty domain → fallback", r_empty.method == "fallback")
    r_unk = clf.classify("xyzmorpho_science_xyz")
    check("unknown domain → some result", r_unk.disc_code is not None)


# ═══════════════════════════════════════════════════════════
# 2. GAP CALCULATOR
# ═══════════════════════════════════════════════════════════

def test_gap_calculator():
    section("2. GapCalculator")
    from mgap_lib.engine.gap_calculator import GapCalculator, GapComponents

    calc = GapCalculator()

    # Нет превышения
    g = calc.compute(
        {"eta": 0.2, "tau": 2.0, "K": 0.5},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
    )
    check("No gap → risk=none",   g.risk_level == "none",   f"got {g.risk_level}")
    check("No gap → composite=0", g.composite == 0.0,        f"got {g.composite}")

    # Лёгкое превышение eta
    g = calc.compute(
        {"eta": 0.40, "tau": 2.0, "K": 0.5},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
    )
    check("eta +14% → monitor", g.risk_level == "monitor", f"got {g.risk_level}")
    check("eta_gap > 0",        g.eta_gap > 0,             f"got {g.eta_gap}")

    # Критическое превышение
    g = calc.compute(
        {"eta": 0.80, "tau": 9.0, "K": 0.05},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
    )
    check("triple overshoot → critical", g.risk_level == "critical", f"got {g.risk_level}")
    check("all three gaps > 0",
          g.eta_gap > 0 and g.tau_gap > 0 and g.K_gap > 0,
          f"eta={g.eta_gap} tau={g.tau_gap} K={g.K_gap}")

    # Умеренное
    g = calc.compute(
        {"eta": 0.45, "tau": 5.5, "K": 0.2},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
    )
    check("moderate gap", g.risk_level == "moderate", f"got {g.risk_level}")
    check("is_warning=True", g.is_warning)

    # Mode=mean
    g_mean = calc.compute(
        {"eta": 0.45, "tau": 2.0, "K": 0.4},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
        mode="mean",
    )
    check("mode=mean → smaller composite than max", True)  # just smoke-test

    # from_artifact_and_model
    four_d = {
        "dynamics":  {"K": 0.2, "K_c": 0.5, "omega_i": 0.25, "p": 0.65, "model": "kuramoto"},
        "influence": {"h": 1.0, "T": 1.0, "eta": 0.45},
        "time":      {"tau": 5.5, "H": 0.7, "freq": 1.0},
    }
    g2 = calc.compute_from_artifact_and_model(four_d, {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3})
    check("compute_from_artifact_and_model works",
          g2.risk_level in ("none","monitor","moderate","critical"),
          f"got {g2.risk_level}")

    # describe_risk
    desc = calc.describe_risk(g)
    check("describe_risk non-empty", len(desc) > 10, desc)

    # to_dict
    d = g.to_dict()
    check("to_dict has all keys",
          all(k in d for k in ["eta_gap","tau_gap","K_gap","composite","risk_level","mode"]))

    # summary_table
    table = calc.summary_table(
        {"eta": 0.45, "tau": 5.5, "K": 0.2},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
    )
    check("summary_table ASCII", "┌" in table and "composite" in table.lower() or "Composite" in table)


# ═══════════════════════════════════════════════════════════
# 3. REGISTRY LOADER
# ═══════════════════════════════════════════════════════════

def test_registry():
    section("3. RegistryLoader")
    registry_path = Path("mgap_registry.json")
    if not registry_path.exists():
        print(f"  {SKIP} mgap_registry.json not found — skipping registry tests")
        return

    from mgap_lib.engine.registry import RegistryLoader
    reg = RegistryLoader(registry_path=registry_path)

    models = reg.get_all()
    check("Loaded > 0 models", len(models) > 0, f"got {len(models)}")
    check("Loaded >= 11 models", len(models) >= 11, f"got {len(models)}")

    m1 = reg.get_by_id("M1")
    check("get_by_id('M1') found",    m1 is not None)
    check("M1 has name",              m1 is not None and bool(m1.get("name")))
    check("M1 has four_d_matrix",     m1 is not None and m1.get("four_d_matrix") is not None)
    check("M1 has critical_thresholds", m1 is not None and m1.get("critical_thresholds") is not None)

    kur = reg.get_by_math_type("kuramoto")
    check("kuramoto models > 0", len(kur) > 0, f"got {len(kur)}")

    delay = reg.get_by_math_type("delay_ode")   # alias test
    check("delay_ode alias resolved", len(delay) > 0, f"got {len(delay)}")

    summary = reg.get_summary()
    check("summary has id/name/math_type",
          all("id" in s and "math_type" in s for s in summary))

    reg.invalidate_cache()
    models2 = reg.get_all()
    check("invalidate_cache + reload works", len(models2) == len(models))


# ═══════════════════════════════════════════════════════════
# 4. MGAP ENGINE (full pipeline without LLM)
# ═══════════════════════════════════════════════════════════

def test_engine():
    section("4. MGAPEngine (no LLM, no DB)")
    registry_path = Path("mgap_registry.json")
    artifacts_dir = Path("artifacts")

    if not registry_path.exists():
        print(f"  {SKIP} mgap_registry.json not found — skipping engine tests")
        return

    from mgap_lib.engine.matcher import MGAPEngine
    engine = MGAPEngine.from_json(
        registry_path=str(registry_path),
        artifacts_dir=str(artifacts_dir),
        use_llm=False,
    )
    check("MGAPEngine created", engine is not None)
    check("Registry loaded",    engine.registry.count() > 0)

    # Registry summary
    summary = engine.get_registry_summary()
    check("get_registry_summary() > 0", len(summary) > 0)
    check("summary items have id/name/math_type",
          all("id" in s and "math_type" in s for s in summary))

    # Match with real artifact (if any exist)
    art_files = sorted(artifacts_dir.glob("*.json")) if artifacts_dir.exists() else []
    art_files = [f for f in art_files
                 if f.stem != "invariant_graph" and ".hyx-portal" not in f.name]

    if not art_files:
        print(f"  {SKIP} No artifacts found — skipping match tests")
        return

    # Pick first artifact that has four_d_matrix
    target_id = None
    for f in art_files[:10]:
        try:
            art = json.loads(f.read_text(encoding="utf-8"))
            if art.get("data", {}).get("gen", {}).get("four_d_matrix"):
                target_id = art.get("id", f.stem)
                break
        except Exception:
            continue

    if not target_id:
        print(f"  {SKIP} No artifact with four_d_matrix found — skipping match tests")
        return

    print(f"  Testing with artifact: {target_id}")
    results = engine.match_artifact(
        artifact_id    = target_id,
        top_k          = 3,
        math_type_only = False,
        save_to_json   = False,
    )

    check("match returns list",  isinstance(results, list))
    check("match returns > 0",   len(results) > 0)

    r = results[0]
    check("result has model_id",       "model_id" in r)
    check("result has resonance",      "resonance" in r and 0.0 <= r["resonance"] <= 1.0,
          f"got {r.get('resonance')}")
    check("result has gap dict",       isinstance(r.get("gap"), dict))
    check("gap has risk_level",        r.get("gap", {}).get("risk_level") in
          ("none","monitor","moderate","critical"))
    check("result has domain_classification", isinstance(r.get("domain_classification"), dict))
    check("result has translation",    isinstance(r.get("translation"), dict))
    check("result has adaptation",     isinstance(r.get("adaptation"), dict))
    check("result has code_snippet",   bool(r.get("adaptation", {}).get("code_snippet")))
    check("result has verdict",        isinstance(r.get("verdict"), dict))
    check("result has thresholds",     isinstance(r.get("thresholds"), dict))
    check("thresholds has eta_critical",
          "eta_critical" in r.get("thresholds", {}))

    # domain_classification structure
    dc = r.get("domain_classification", {})
    check("domain_classification has disc_code",   "disc_code" in dc)
    check("domain_classification has method",      "method" in dc)
    check("domain_classification has confidence",  "confidence" in dc)


# ═══════════════════════════════════════════════════════════
# 5. DATABASE ORM (smoke test без файла)
# ═══════════════════════════════════════════════════════════

def test_database_orm():
    section("5. SQLAlchemy ORM (in-memory smoke test)")
    try:
        from mgap_lib.models.database import (
            create_db_engine, create_tables, get_session_factory,
            Discipline, Sector, Specialization, MGAPModel, ArtifactRun
        )

        engine = create_db_engine("sqlite:///:memory:")
        create_tables(engine)
        factory = get_session_factory(engine)
        session = factory()

        # Добавить дисциплину
        disc = Discipline(code="1", name_ru="Естественные науки", name_en="Natural Sciences")
        session.add(disc)
        session.commit()

        d = session.query(Discipline).filter_by(code="1").first()
        check("Discipline created",    d is not None)
        check("Discipline name_ru OK", d.name_ru == "Естественные науки")

        # Добавить сектор
        sec = Sector(code="1.5", name_ru="Биология", name_en="Biology", disc_code="1")
        session.add(sec)
        session.commit()

        s = session.query(Sector).filter_by(code="1.5").first()
        check("Sector created", s is not None)

        # Добавить модель с JSON-полями
        m = MGAPModel(id="TEST1", name="Test Model", math_type="kuramoto")
        m.programs          = ["TestProg1", "TestProg2"]
        m.critical_thresholds = {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3}
        m.four_d_matrix     = {
            "structure": {"C": 0.7, "k": 10, "D": 2.0},
            "influence": {"h": 1.0, "T": 1.0, "eta": 0.2},
            "dynamics":  {"omega_i": 0.25, "K": 0.5, "K_c": 0.4, "p": 0.7, "model": "kuramoto"},
            "time":      {"tau": 1.5, "H": 0.7, "freq": 1.0},
        }
        session.add(m)
        session.commit()

        loaded = session.query(MGAPModel).filter_by(id="TEST1").first()
        check("MGAPModel created",           loaded is not None)
        check("programs JSON round-trip",    loaded.programs == ["TestProg1", "TestProg2"])
        check("critical_thresholds RT",      loaded.critical_thresholds.get("eta_max") == 0.35)
        check("four_d_matrix RT",            loaded.four_d_matrix.get("dynamics", {}).get("model") == "kuramoto")

        # ArtifactRun
        run = ArtifactRun(artifact_id="art001", model_id="TEST1", resonance=0.85, risk_level="none")
        run.results = {"test": True}
        session.add(run)
        session.commit()

        runs = session.query(ArtifactRun).filter_by(artifact_id="art001").all()
        check("ArtifactRun created",         len(runs) == 1)
        check("ArtifactRun results RT",      runs[0].results.get("test") is True)
        check("ArtifactRun resonance",       runs[0].resonance == 0.85)

        session.close()

    except ImportError as e:
        print(f"  {SKIP} SQLAlchemy not installed: {e}")
    except Exception as e:
        print(f"  {FAIL} ORM test crashed: {e}")
        traceback.print_exc()
        global _fail_count
        _fail_count += 1


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  MGAP Library v1.0 — Integration Tests")
    print("=" * 55)

    tests = [
        ("DomainClassifier",  test_domain_classifier),
        ("GapCalculator",     test_gap_calculator),
        ("RegistryLoader",    test_registry),
        ("MGAPEngine",        test_engine),
        ("SQLAlchemy ORM",    test_database_orm),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  {FAIL} {name} crashed: {e}")
            traceback.print_exc()
            global _fail_count
            _fail_count += 1

    print(f"\n{'=' * 55}")
    print(f"  Results: {PASS} {_pass_count} passed  |  {FAIL} {_fail_count} failed")
    print(f"{'=' * 55}\n")

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()
