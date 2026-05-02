#!/usr/bin/env python3
# mgap_lib/cli/mgap_cli.py — MGAP Library v1.0
"""
CLI для MGAP Library.

Команды:
  match       — матч одного артефакта
  batch       — все артефакты в папке
  registry    — вывод реестра
  classify    — классификация домена
  stats       — статистика реестра
  init-db     — инициализация БД (ярлык для scripts/init_db.py)

Использование:
  python mgap_lib/cli/mgap_cli.py match --artifact 32d4aa917ac4
  python mgap_lib/cli/mgap_cli.py match --artifact 32d4aa917ac4 --top_k 5 --all_types
  python mgap_lib/cli/mgap_cli.py batch --min_res 0.4
  python mgap_lib/cli/mgap_cli.py classify --domain biology
  python mgap_lib/cli/mgap_cli.py registry --math_type kuramoto
  python mgap_lib/cli/mgap_cli.py stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Путь к корню проекта (2 уровня вверх от cli/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MGAP.cli")


def _make_engine(args):
    from mgap_lib.engine.matcher import MGAPEngine
    return MGAPEngine.from_json(
        registry_path = getattr(args, "registry", "mgap_registry.json"),
        artifacts_dir = getattr(args, "artifacts_dir", "artifacts"),
        use_llm       = not getattr(args, "no_llm", False),
        gap_mode      = getattr(args, "gap_mode", "max"),
    )


def cmd_match(args):
    engine  = _make_engine(args)
    results = engine.match_artifact(
        artifact_id    = args.artifact,
        top_k          = args.top_k,
        math_type_only = not args.all_types,
        model_id       = args.model or None,
        sector_filter  = args.sector or None,
        save_to_json   = args.save,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_batch(args):
    engine  = _make_engine(args)
    results = engine.match_batch(
        top_k         = args.top_k,
        math_type_only = not args.all_types,
        min_resonance  = args.min_res,
        save_to_db     = False,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_registry(args):
    from mgap_lib.engine.registry import RegistryLoader
    from mgap_lib.engine.matcher import _norm_mt
    registry = RegistryLoader(registry_path=Path(args.registry))
    models   = registry.get_all()
    if args.math_type:
        models = [m for m in models if _norm_mt(m.get("math_type","")) == _norm_mt(args.math_type)]
    if args.sector:
        models = [m for m in models if m.get("sector_code") == args.sector]
    summary  = [
        {
            "id":        m.get("id"),
            "name":      m.get("name"),
            "logia":     m.get("logia"),
            "math_type": m.get("math_type"),
            "disc_code": m.get("disc_code"),
            "sector_code": m.get("sector_code"),
            "programs":  m.get("programs", []),
        }
        for m in models
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_classify(args):
    from mgap_lib.engine.domain_classifier import DomainClassifier
    clf = DomainClassifier()
    if args.domain:
        print(clf.describe(args.domain))
        if args.verbose:
            print(json.dumps(clf.classify(args.domain).to_dict(), ensure_ascii=False, indent=2))
    elif args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
        for d in domains:
            print(f"  {d:30s} → {clf.describe(d)}")
    else:
        print("Укажите --domain или --domains")


def cmd_stats(args):
    from mgap_lib.engine.registry import RegistryLoader
    registry = RegistryLoader(registry_path=Path(args.registry))
    models   = registry.get_all()
    mt_count: dict = {}
    sc_count: dict = {}
    lo_count: dict = {}
    for m in models:
        mt = m.get("math_type", "unknown")
        sc = m.get("sector_code") or "—"
        lo = m.get("logia") or "—"
        mt_count[mt] = mt_count.get(mt, 0) + 1
        sc_count[sc] = sc_count.get(sc, 0) + 1
        lo_count[lo] = lo_count.get(lo, 0) + 1

    print("\n=== MGAP Registry Stats ===")
    print(f"Total models: {len(models)}")
    print("\nBy math_type:")
    for k, v in sorted(mt_count.items()):
        print(f"  {k:20s} {v}")
    print("\nBy logia:")
    for k, v in sorted(lo_count.items()):
        print(f"  {k:25s} {v}")
    print("\nBy sector_code:")
    for k, v in sorted(sc_count.items()):
        print(f"  {k:10s} {v}")

    # GAP calculator demo
    from mgap_lib.engine.gap_calculator import GapCalculator
    calc = GapCalculator()
    demo_gap = calc.compute(
        {"eta": 0.45, "tau": 5.5, "K": 0.2},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
    )
    print(f"\nGap calculator demo (eta=0.45/0.35, tau=5.5/4.5, K=0.2/0.3):")
    print(calc.summary_table(
        {"eta": 0.45, "tau": 5.5, "K": 0.2},
        {"eta_max": 0.35, "tau_max": 4.5, "K_min": 0.3},
    ))
    print(f"Risk: {demo_gap.risk_level} | {calc.describe_risk(demo_gap)}")


def cmd_init_db(args):
    from mgap_lib.scripts.init_db import main as init_main
    # Подменяем sys.argv
    sys.argv = [
        "init_db.py",
        f"--db={args.db}",
        f"--registry={args.registry}",
    ]
    if args.reset:
        sys.argv.append("--reset")
    init_main()


def main():
    parser = argparse.ArgumentParser(
        description="HX-AM MGAP Library v1.0 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Общие аргументы
    def add_common(p):
        p.add_argument("--registry",     default="mgap_registry.json")
        p.add_argument("--artifacts-dir", default="artifacts", dest="artifacts_dir")
        p.add_argument("--no-llm",        action="store_true", dest="no_llm")
        p.add_argument("--gap-mode",      default="max", choices=["max","mean","rms"], dest="gap_mode")

    # match
    p_match = sub.add_parser("match", help="Матч одного артефакта")
    add_common(p_match)
    p_match.add_argument("--artifact",  required=True)
    p_match.add_argument("--top_k",     type=int, default=3)
    p_match.add_argument("--all_types", action="store_true")
    p_match.add_argument("--model",     default="")
    p_match.add_argument("--sector",    default="")
    p_match.add_argument("--save",      action="store_true")

    # batch
    p_batch = sub.add_parser("batch", help="Batch: все артефакты")
    add_common(p_batch)
    p_batch.add_argument("--top_k",     type=int, default=2)
    p_batch.add_argument("--all_types", action="store_true")
    p_batch.add_argument("--min_res",   type=float, default=0.3)

    # registry
    p_reg = sub.add_parser("registry", help="Список моделей реестра")
    add_common(p_reg)
    p_reg.add_argument("--math_type", default="")
    p_reg.add_argument("--sector",    default="")

    # classify
    p_clf = sub.add_parser("classify", help="Классификация домена → UNESCO")
    p_clf.add_argument("--domain",  default="")
    p_clf.add_argument("--domains", default="", help="Через запятую")
    p_clf.add_argument("--verbose", action="store_true")

    # stats
    p_stats = sub.add_parser("stats", help="Статистика")
    add_common(p_stats)

    # init-db
    p_db = sub.add_parser("init-db", help="Инициализация БД")
    p_db.add_argument("--db",       default="sqlite:///mgap.db")
    p_db.add_argument("--registry", default="mgap_registry.json")
    p_db.add_argument("--reset",    action="store_true")

    args = parser.parse_args()

    dispatch = {
        "match":    cmd_match,
        "batch":    cmd_batch,
        "registry": cmd_registry,
        "classify": cmd_classify,
        "stats":    cmd_stats,
        "init-db":  cmd_init_db,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
