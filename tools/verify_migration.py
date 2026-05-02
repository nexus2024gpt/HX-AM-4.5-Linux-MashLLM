# tools/verify_migration.py — HX-AM v4.2
"""
Проверяет качество миграции архива.
Показывает: сколько артефактов имеют four_d_matrix, какие пропущены, валидны ли данные.

CLI:
  python tools/verify_migration.py           # полный отчёт
  python tools/verify_migration.py --fix     # исправить явные ошибки (диапазоны)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_artifacts():
    arts_dir = Path("artifacts")
    if not arts_dir.exists():
        print("artifacts/ не найден")
        return

    total = 0
    with_4d = 0
    without_4d = []
    invalid_4d = []
    portals_total = 0
    portals_with_4d = 0
    portals_without = []

    for f in sorted(arts_dir.glob("*.json")):
        if f.stem == "invariant_graph":
            continue

        is_portal = ".hyx-portal" in f.name

        try:
            art = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            print(f"  ⚠ cannot read {f.name}")
            continue

        if is_portal:
            portals_total += 1
            if art.get("four_d_matrix"):
                portals_with_4d += 1
                # Валидируем
                err = validate_four_d(art["four_d_matrix"], f.name)
                if err:
                    invalid_4d.append({"file": f.name, "errors": err})
            else:
                portals_without.append(f.name)
        else:
            total += 1
            gen = art.get("data", {}).get("gen", {})
            four_d = gen.get("four_d_matrix")
            if four_d:
                with_4d += 1
                err = validate_four_d(four_d, f.name)
                if err:
                    invalid_4d.append({"file": f.name, "errors": err})
            else:
                without_4d.append({
                    "file": f.name,
                    "id": art.get("id", f.stem),
                    "domain": art.get("data", {}).get("domain", "?"),
                })

    # ── Граф ─────────────────────────────────────────────────────
    graph_patched = False
    graph_path = arts_dir / "invariant_graph.json"
    if graph_path.exists():
        try:
            g = json.loads(graph_path.read_text())
            graph_patched = "_v42_graph_patch" in g
        except Exception:
            pass

    # ── 4D индекс ────────────────────────────────────────────────
    four_d_idx = arts_dir / "four_d_index.jsonl"
    idx_count = 0
    if four_d_idx.exists():
        idx_count = sum(1 for line in four_d_idx.read_text().splitlines() if line.strip())

    # ── Симуляции ─────────────────────────────────────────────────
    sim_dir = Path("sim_results")
    sim_count = len(list(sim_dir.glob("*.json"))) if sim_dir.exists() else 0

    # ── Отчёт ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  HX-AM v4.2 Migration Verification Report")
    print("=" * 55)

    pct = (with_4d / total * 100) if total > 0 else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"\n  Обычные артефакты:  [{bar}] {with_4d}/{total} ({pct:.0f}%) имеют four_d_matrix")

    ppct = (portals_with_4d / portals_total * 100) if portals_total > 0 else 0
    pbar = "█" * int(ppct / 5) + "░" * (20 - int(ppct / 5))
    print(f"  hyx-portal файлы:   [{pbar}] {portals_with_4d}/{portals_total} ({ppct:.0f}%) имеют four_d_matrix")

    print(f"\n  invariant_graph.json патч v4.2: {'✅ да' if graph_patched else '❌ нет (запусти patch_graph.py)'}")
    print(f"  four_d_index.jsonl:             {idx_count} записей {'✅' if idx_count > 0 else '⚠ пустой'}")
    print(f"  sim_results/:                   {sim_count} стресс-тестов")

    if without_4d:
        print(f"\n  ⚠ Артефакты БЕЗ four_d_matrix ({len(without_4d)}):")
        for item in without_4d[:10]:
            print(f"    · {item['id']}  [{item['domain']}]  {item['file']}")
        if len(without_4d) > 10:
            print(f"    ... и ещё {len(without_4d)-10}")

    if portals_without:
        print(f"\n  ⚠ Portal-файлы БЕЗ four_d_matrix ({len(portals_without)}):")
        for name in portals_without[:5]:
            print(f"    · {name}")

    if invalid_4d:
        print(f"\n  ❌ Артефакты с НЕВАЛИДНЫМ four_d_matrix ({len(invalid_4d)}):")
        for item in invalid_4d[:5]:
            print(f"    · {item['file']}: {', '.join(item['errors'][:3])}")

    if not without_4d and not portals_without and not invalid_4d:
        print("\n  ✅ Все артефакты успешно мигрированы!")

    print()

    return {
        "total": total,
        "with_4d": with_4d,
        "without_4d": len(without_4d),
        "portals_total": portals_total,
        "portals_with_4d": portals_with_4d,
        "invalid_4d": len(invalid_4d),
        "graph_patched": graph_patched,
        "idx_count": idx_count,
        "sim_count": sim_count,
    }


def validate_four_d(fd: dict, filename: str) -> list:
    """Проверяет four_d_matrix на корректность диапазонов."""
    errors = []
    if not isinstance(fd, dict):
        return ["not a dict"]

    RANGES = {
        ("structure", "C"): (0.0, 1.0),
        ("structure", "k"): (1.0, 50.0),
        ("structure", "D"): (1.0, 4.0),
        ("influence", "h"): (0.0, 5.0),
        ("influence", "T"): (0.0, 5.0),
        ("influence", "eta"): (0.0, 1.0),
        ("dynamics", "omega_i"): (0.0, 5.0),
        ("dynamics", "K"): (0.0, 2.0),
        ("dynamics", "K_c"): (0.0, 2.0),
        ("dynamics", "p"): (0.0, 1.0),
        ("time", "tau"): (0.0, 20.0),
        ("time", "H"): (0.0, 1.0),
        ("time", "freq"): (0.0, 10.0),
    }

    for (layer, param), (lo, hi) in RANGES.items():
        val = fd.get(layer, {})
        if not isinstance(val, dict):
            errors.append(f"{layer} not dict")
            continue
        v = val.get(param)
        if v is None:
            errors.append(f"{layer}.{param} missing")
        else:
            try:
                fv = float(v)
                if not (lo <= fv <= hi):
                    errors.append(f"{layer}.{param}={fv} out of [{lo},{hi}]")
            except (ValueError, TypeError):
                errors.append(f"{layer}.{param}='{v}' not float")

    model = fd.get("dynamics", {}).get("model", "")
    KNOWN_MODELS = {"kuramoto", "percolation", "ising", "delay", "lotka_volterra", "graph_invariant", "fram", "coleman"}
    if model not in KNOWN_MODELS:
        errors.append(f"dynamics.model='{model}' unknown")

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Исправить невалидные значения (зажать в диапазон)")
    args = parser.parse_args()

    check_artifacts()
