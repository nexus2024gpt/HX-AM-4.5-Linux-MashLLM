# mgap_lib/engine/dimensional_normalizer.py — HX-AM v4.7
"""
Слой размерной нормализации (П-теорема Бэкингема, адаптация).

Проблема, которую решает модуль:
  FourDBuilder генерирует tau в безразмерных единицах [0, 20].
  Модели в реестре описывают реальные физические системы с конкретными
  временными масштабами (инкубационный период в днях, орбитальный период
  в годах, время релаксации в миллисекундах). Прямое сравнение
  tau_abstract с tau_max из critical_thresholds бессмысленно без
  размерного перевода.

Решение:
  Каждая модель декларирует dimensional_basis с t0_value/t0_unit.
  tau_real = tau_abstract * t0_value → проверяется против tau_physical_range.

Обратная совместимость:
  Если dimensional_basis отсутствует → нормализатор возвращает
  tau_abstract без изменений и физическую проверку не выполняет.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("HXAM.mgap.dimensional")


class DimensionalNormalizer:
    """
    Переводит безразмерные параметры 4D-матрицы в физические единицы
    конкретной отраслевой модели.

    Пример:
        n = DimensionalNormalizer()
        model = registry.get_by_id("M08")   # эпидемиология
        tau_real, unit = n.normalize_tau(5.0, model)
        # → (5.0, "days")   (t0=1 день по умолчанию)
        # при t0=5: tau_real=25.0 дней
    """

    def normalize_tau(
        self,
        tau_abstract: float,
        model: Dict,
    ) -> Tuple[float, str]:
        """
        Переводит безразмерный tau в физические единицы модели.

        Returns:
            (tau_real, unit_string)
        """
        basis = model.get("dimensional_basis") or {}
        t0    = float(basis.get("t0_value", 1.0))
        unit  = str(basis.get("t0_unit", "abstract"))
        return round(tau_abstract * t0, 4), unit

    def normalize_eta(
        self,
        eta_abstract: float,
        model: Dict,
    ) -> Tuple[float, str]:
        """
        eta [0,1] — уже безразмерна, возвращаем с меткой домена.
        Нормализация нужна для отображения в UI, не для вычислений.
        """
        basis     = model.get("dimensional_basis") or {}
        eta_label = str(basis.get("eta_unit", "dimensionless"))
        return round(eta_abstract, 4), eta_label

    def normalize_K(
        self,
        K_abstract: float,
        model: Dict,
    ) -> Tuple[float, str]:
        """
        K [0,2] — безразмерная константа связи.
        Возвращаем с физической интерпретацией из модели.
        """
        basis   = model.get("dimensional_basis") or {}
        K_label = str(basis.get("K_unit", "coupling_constant"))
        return round(K_abstract, 4), K_label

    def check_physical_range(
        self,
        tau_real: float,
        model: Dict,
    ) -> bool:
        """
        Проверяет, попадает ли физическое tau в декларированный диапазон модели.

        Returns:
            True  — tau в допустимом диапазоне (или диапазон не задан)
            False — tau выходит за физические границы модели
        """
        basis = model.get("dimensional_basis") or {}
        phys_range = basis.get("tau_physical_range")
        if not phys_range or len(phys_range) != 2:
            return True     # нет ограничений → не блокируем
        lo, hi = float(phys_range[0]), float(phys_range[1])
        return lo <= tau_real <= hi

    def get_physical_interpretation(
        self,
        flat: Dict,
        model: Dict,
    ) -> Dict:
        """
        Возвращает полный блок физической интерпретации для матч-результата.

        flat: _flat_4d(four_d_matrix) — {"tau": ..., "K": ..., "eta": ...}
        model: модель из реестра

        Returns:
            {
                "tau_abstract": float,
                "tau_real": float,
                "tau_unit": str,
                "in_physical_range": bool,
                "warning": str | None,
                "eta_abstract": float,
                "eta_unit": str,
                "K_abstract": float,
                "K_unit": str,
                "basis_available": bool,
            }
        """
        basis_available = bool(model.get("dimensional_basis"))

        tau_abstract = float(flat.get("tau", 0.5))
        tau_real, tau_unit = self.normalize_tau(tau_abstract, model)
        in_range = self.check_physical_range(tau_real, model)

        basis = model.get("dimensional_basis") or {}
        phys_range = basis.get("tau_physical_range")

        warning: Optional[str] = None
        if not in_range and phys_range:
            lo, hi = phys_range[0], phys_range[1]
            direction = "ниже минимума" if tau_real < lo else "выше максимума"
            warning = (
                f"τ={tau_real:.3f} {tau_unit} {direction} "
                f"физически допустимого диапазона [{lo}, {hi}] "
                f"для модели «{model.get('name', '?')}»"
            )
            logger.debug(f"DimensionalNormalizer: {warning}")

        eta_abstract = float(flat.get("eta", 0.2))
        eta_real, eta_unit = self.normalize_eta(eta_abstract, model)

        K_abstract = float(flat.get("K", 0.35))
        K_real, K_unit = self.normalize_K(K_abstract, model)

        return {
            "tau_abstract":      tau_abstract,
            "tau_real":          tau_real,
            "tau_unit":          tau_unit,
            "in_physical_range": in_range,
            "warning":           warning,
            "eta_abstract":      eta_abstract,
            "eta_unit":          eta_unit,
            "K_abstract":        K_abstract,
            "K_unit":            K_unit,
            "basis_available":   basis_available,
        }
