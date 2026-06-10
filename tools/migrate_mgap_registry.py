#!/usr/bin/env python3
# tools/migrate_mgap_registry.py — HX-AM v4.7
"""
Безопасная миграция mgap_registry.json:
  - Добавляет блок dimensional_basis к каждой модели
  - Добавляет _dynamic в critical_thresholds (только для percolation и kuramoto)

Стратегия:
  - Существующие поля не изменяются и не удаляются
  - Новые поля добавляются только если их нет
  - Источник данных: logia + math_type модели
  - Результат: mgap_registry_v2.json (оригинал не перезаписывается до ручной проверки)

CLI:
  python tools/migrate_mgap_registry.py --dry-run   # показать что изменится
  python tools/migrate_mgap_registry.py             # создать mgap_registry_v2.json
  python tools/migrate_mgap_registry.py --apply     # заменить оригинал (после проверки)
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

REGISTRY_PATH = Path("mgap_registry.json")
OUTPUT_PATH   = Path("mgap_registry_v2.json")

# ── Размерные базисы по логии ─────────────────────────────────────────────────
# t0_value: во сколько раз безразмерная единица tau соответствует физической единице
# tau_physical_range: допустимый физический диапазон tau в t0_unit
LOGIA_DIMENSIONAL: dict = {
    "Физика": {
        "t0_value": 1.0, "t0_unit": "abstract_time",
        "tau_physical_range": [0.1, 20.0],
        "eta_unit": "noise_fraction",
        "K_unit": "coupling_constant",
    },
    "Химия": {
        "t0_value": 1.0, "t0_unit": "reaction_times",
        "tau_physical_range": [0.05, 10.0],
        "eta_unit": "concentration_variance",
        "K_unit": "rate_constant_ratio",
    },
    "Биология": {
        "t0_value": 1.0, "t0_unit": "generations",
        "tau_physical_range": [0.5, 50.0],
        "eta_unit": "fitness_noise",
        "K_unit": "interaction_strength",
    },
    "Экология": {
        "t0_value": 1.0, "t0_unit": "years",
        "tau_physical_range": [0.5, 100.0],
        "eta_unit": "environmental_variance",
        "K_unit": "competition_coefficient",
    },
    "Экономика": {
        "t0_value": 0.25, "t0_unit": "quarters",
        "tau_physical_range": [0.25, 20.0],
        "eta_unit": "volatility_index",
        "K_unit": "market_coupling",
    },
    "Логистика": {
        "t0_value": 1.0, "t0_unit": "days",
        "tau_physical_range": [0.5, 90.0],
        "eta_unit": "demand_cv",
        "K_unit": "network_connectivity",
    },
    "Социология": {
        "t0_value": 30.0, "t0_unit": "days",
        "tau_physical_range": [7.0, 730.0],
        "eta_unit": "opinion_variance",
        "K_unit": "social_influence",
    },
    "Политология": {
        "t0_value": 30.0, "t0_unit": "days",
        "tau_physical_range": [30.0, 1825.0],
        "eta_unit": "instability_index",
        "K_unit": "institutional_coupling",
    },
    "Геонауки": {
        "t0_value": 1000.0, "t0_unit": "years",
        "tau_physical_range": [100.0, 20000.0],
        "eta_unit": "stress_heterogeneity",
        "K_unit": "tectonic_coupling",
    },
    "Океанография": {
        "t0_value": 10.0, "t0_unit": "years",
        "tau_physical_range": [1.0, 500.0],
        "eta_unit": "current_variance",
        "K_unit": "circulation_strength",
    },
    "Технологии": {
        "t0_value": 0.001, "t0_unit": "seconds",
        "tau_physical_range": [0.0001, 10.0],
        "eta_unit": "signal_noise_ratio",
        "K_unit": "synchronization_gain",
    },
    "Инженерия": {
        "t0_value": 1.0, "t0_unit": "hours",
        "tau_physical_range": [0.1, 720.0],
        "eta_unit": "load_variance",
        "K_unit": "structural_coupling",
    },
    "Материаловедение": {
        "t0_value": 1.0, "t0_unit": "relaxation_times",
        "tau_physical_range": [0.01, 100.0],
        "eta_unit": "disorder_fraction",
        "K_unit": "exchange_integral_ratio",
    },
    "Астрономия": {
        "t0_value": 365.25, "t0_unit": "days",
        "tau_physical_range": [10.0, 100000.0],
        "eta_unit": "orbital_perturbation",
        "K_unit": "gravitational_coupling",
    },
    "Междисциплинарно": {
        "t0_value": 1.0, "t0_unit": "abstract_units",
        "tau_physical_range": [0.1, 100.0],
        "eta_unit": "dimensionless",
        "K_unit": "coupling_constant",
    },
}

_DEFAULT_BASIS = {
    "t0_value": 1.0, "t0_unit": "abstract_units",
    "tau_physical_range": [0.1, 20.0],
    "eta_unit": "dimensionless",
    "K_unit": "coupling_constant",
}

# ── Динамические пороги по math_type ─────────────────────────────────────────
DYNAMIC_BY_MATH_TYPE: dict = {
    "percolation": {
        "p_crit": {
            "formula": "1 / mean_k",
            "topology_conditions": {
                "erdos_renyi":  "1 / mean_k",
                "scale_free":   "mean_k / (mean_k_sq - mean_k)",
                "regular_grid": 0.5927,
                "small_world":  "1 / (mean_k * (1 - clustering_coef))",
            },
            "required_inputs": ["mean_k"],
            "fallback_field":  "p_crit",
        }
    },
    "kuramoto": {
        "eta_max": {
            "formula": "K_c * 0.6",
            "required_inputs": ["K_c"],
            "fallback_field":  "eta_max",
        }
    },
}


def build_dimensional_basis(model: dict) -> dict:
    """Строит блок dimensional_basis для модели."""
    logia = model.get("logia", "")
    basis = dict(LOGIA_DIMENSIONAL.get(logia, _DEFAULT_BASIS))
    # Добавляем meta
    basis["_source"] = "migrate_mgap_registry.py v4.7"
    basis["_note"]   = (
        "Размерный базис: tau_abstract * t0_value = tau_real в t0_unit. "
        "tau_physical_range — допустимый физический диапазон для этой модели."
    )
    return basis


def build_dynamic_thresholds(model: dict, existing_ct: dict) -> dict | None:
    """
    Строит блок _dynamic для critical_thresholds.
    Возвращает None если math_type не поддерживается.
    """
    mt = str(model.get("math_type", "")).lower().strip()
    if mt in ("delay_ode", "delay-ode"):
        mt = "delay"

    template = DYNAMIC_BY_MATH_TYPE.get(mt)
    if not template:
        return None

    dynamic = {}
    for param, spec in template.items():
        # Подставляем fallback из существующих critical_thresholds
        fallback_field = spec.get("fallback_field", param)
        fallback_val   = existing_ct.get(fallback_field)
        entry          = dict(spec)
        if fallback_val is not None:
            entry["fallback"] = fallback_val
        entry.pop("fallback_field", None)
        dynamic[param] = entry

    return dynamic


def migrate_model(model: dict, dry_run: bool = False) -> tuple[dict, list[str]]:
    """
    Мигрирует одну модель. Возвращает (обновлённая_модель, список_изменений).
    Существующие поля не перезаписываются.
    """
    changes = []
    m = dict(model)

    # 1. dimensional_basis
    if "dimensional_basis" not in m:
        if not dry_run:
            m["dimensional_basis"] = build_dimensional_basis(m)
        changes.append(f"+ dimensional_basis (logia={m.get('logia', '?')})")

    # 2. _dynamic в critical_thresholds
    ct = m.get("critical_thresholds", {})
    if "_dynamic" not in ct:
        dynamic = build_dynamic_thresholds(m, ct)
        if dynamic:
            if not dry_run:
                ct_new = dict(ct)
                ct_new["_dynamic"] = dynamic
                m["critical_thresholds"] = ct_new
            changes.append(
                f"+ critical_thresholds._dynamic "
                f"(params: {list(dynamic.keys())})"
            )

    return m, changes


def main():
    parser = argparse.ArgumentParser(description="HX-AM v4.7 — MGAP Registry Migration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать изменения без создания файла")
    parser.add_argument("--apply",   action="store_true",
                        help="Заменить оригинальный файл (только после проверки v2)")
    args = parser.parse_args()

    if not REGISTRY_PATH.exists():
        print(f"❌ {REGISTRY_PATH} не найден")
        return

    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    models = data.get("models", [])

    print(f"\n🔄 HX-AM v4.7 — MGAP Registry Migration")
    print(f"   Источник: {REGISTRY_PATH}")
    print(f"   Моделей: {len(models)}")
    print(f"   Режим: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    migrated  = []
    all_changes: list[str] = []

    for model in models:
        m_id = model.get("id", "?")
        updated, changes = migrate_model(model, dry_run=args.dry_run)
        migrated.append(updated)
        if changes:
            print(f"  [{m_id}] {model.get('name', '')[:40]}")
            for ch in changes:
                print(f"      {ch}")
            all_changes.extend(changes)

    print(f"\n  Итого изменений: {len(all_changes)}")

    if args.dry_run:
        print("  (dry-run: файл не создан)")
        return

    # Создаём v2
    data_v2 = dict(data)
    data_v2["models"] = migrated
    data_v2["_migration"] = {
        "version": "v4.7",
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "changes_count": len(all_changes),
        "script": "tools/migrate_mgap_registry.py",
    }

    OUTPUT_PATH.write_text(
        json.dumps(data_v2, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n✅ Создан: {OUTPUT_PATH}")

    if args.apply:
        backup = REGISTRY_PATH.with_suffix(
            f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.json"
        )
        shutil.copy2(REGISTRY_PATH, backup)
        print(f"💾 Бэкап: {backup}")
        OUTPUT_PATH.replace(REGISTRY_PATH)
        print(f"✅ {REGISTRY_PATH} обновлён")
    else:
        print(f"   Проверь {OUTPUT_PATH}, затем запусти с --apply")


if __name__ == "__main__":
    main()
