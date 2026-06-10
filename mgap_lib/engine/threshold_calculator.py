# mgap_lib/engine/threshold_calculator.py — HX-AM v4.7
"""
Динамический вычислитель критических порогов.

Проблема:
  В mgap_registry.json пороги зашиты как статические float-константы
  (например, p_crit=0.409). Истинный порог зависит от топологии сети:
    ER:         p_crit = 1 / mean_k
    Scale-Free: p_crit = mean_k / (mean_k_sq - mean_k)  [Molloy-Reed]
    2D Grid:    p_crit ≈ 0.5927  [точное]

Решение:
  Каждая модель может дополнительно содержать блок _dynamic в
  critical_thresholds с декларативными формулами. ThresholdCalculator
  вычисляет актуальные пороги из структурного контекста артефакта.
  Если контекст не предоставлен → возвращает статический fallback.

БЕЗОПАСНОСТЬ eval():
  Формулы из реестра выполняются через ограниченный eval с двойной
  защитой:
    1. Регулярное выражение: разрешены только цифры, арифметические
       знаки, пробелы и токены из required_inputs.
    2. Словарь переменных: передаётся только явный whitelist,
       __builtins__ = {} — стандартные функции недоступны.
  Это предотвращает инъекции даже при полуавтоматической генерации
  или внешних правках реестра.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("HXAM.mgap.thresholds")

# Разрешённые символы в формулах: цифры, точка, арифметика, пробелы, скобки.
# Имена переменных добавляются динамически из required_inputs.
_SAFE_FORMULA_RE_BASE = re.compile(
    r"^[\d\s\.\+\-\*/\(\)]+$"
)

# Разрешённые имена переменных (токены, не числа)
_VALID_TOKEN_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _validate_formula(formula: str, allowed_tokens: Set[str]) -> bool:
    """
    Двойная валидация формулы перед eval():

    1. Убеждаемся, что строка содержит только:
       - Цифры и точки (числа)
       - Арифметические операторы: + - * / ( )
       - Пробелы
       - Токены из allowed_tokens (имена переменных)

    2. Проверяем отсутствие потенциально опасных конструкций:
       - Ключевые слова Python (import, exec, eval, etc.)
       - Двойное подчёркивание (__dunder__)
       - Квадратные скобки, точки (доступ к атрибутам)

    Returns:
        True  — формула безопасна для eval
        False — формула содержит недопустимые конструкции
    """
    if not formula or len(formula) > 200:
        return False

    # Блокируем опасные паттерны независимо от токенов
    dangerous_patterns = [
        "__", "import", "exec", "eval", "open", "os.", "sys.",
        "getattr", "setattr", "globals", "locals", "[", "]",
        "lambda", "class", "def ", "return", "yield", "raise",
    ]
    # Примечание: одиночная точка '.' НЕ блокируется — она используется
    # в числах (0.5927). Составные паттерны 'os.' и 'sys.' уже выше.
    formula_lower = formula.lower()
    for pat in dangerous_patterns:
        if pat in formula_lower:
            logger.warning(f"ThresholdCalculator: dangerous pattern '{pat}' in formula: {formula!r}")
            return False

    # Строим расширенный паттерн: базовые символы + разрешённые токены
    token_alts = "|".join(re.escape(t) for t in sorted(allowed_tokens, key=len, reverse=True))
    # Разрешённая строка: числа, операторы, пробелы, разрешённые имена переменных
    combined = re.compile(
        rf"^([\d\s\.\+\-\*/\(\)]|({token_alts}))*$"
    )
    if not combined.match(formula):
        logger.warning(f"ThresholdCalculator: formula failed character whitelist: {formula!r}")
        return False

    return True


class ThresholdCalculator:
    """
    Вычисляет критические пороги модели с учётом структурного контекста.

    structural_context (опциональный):
        {
            "mean_k":          float,   # средняя степень узла из four_d.structure.k
            "mean_k_sq":       float,   # <k²> — приближение из k и D
            "topology_type":   str,     # erdos_renyi | scale_free | small_world | regular_grid
            "N":               int,     # размер сети (если известен)
            "clustering_coef": float,   # C из four_d.structure
        }

    Пример:
        calc = ThresholdCalculator()

        # Только статика (нет контекста):
        result = calc.compute(model)
        # → {"p_crit": 0.409, "eta_max": 0.5, ...}

        # С динамикой (ER-граф, k=13.6):
        ctx = {"mean_k": 13.6, "topology_type": "erdos_renyi"}
        result = calc.compute(model, ctx)
        # → {"p_crit": 0.0735, "_p_crit_dynamic": True, "eta_max": 0.5, ...}
    """

    def compute(
        self,
        model: Dict,
        structural_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает актуальные пороги (статические + динамические override).

        Поля с префиксом _ (например, _dynamic) не включаются в результат.
        Для каждого динамически вычисленного поля добавляется маркер
        _{field}_dynamic = True для UI.
        """
        static = model.get("critical_thresholds", {})

        # Отфильтровываем служебные поля
        result: Dict[str, Any] = {
            k: v for k, v in static.items()
            if not k.startswith("_")
        }

        dynamic_spec = static.get("_dynamic", {})
        if not dynamic_spec or not structural_context:
            return result

        # Обогащаем контекст приближениями если не хватает данных
        ctx = self._enrich_context(structural_context, model)

        for param, spec in dynamic_spec.items():
            if not isinstance(spec, dict):
                continue
            value = self._compute_param(param, spec, ctx)
            if value is not None:
                old_val = result.get(param)
                result[param] = value
                result[f"_{param}_dynamic"] = True
                logger.debug(
                    f"ThresholdCalculator: {param} "
                    f"static={old_val} → dynamic={value:.4f}"
                )

        return result

    def _compute_param(
        self,
        param: str,
        spec: Dict,
        ctx: Dict,
    ) -> Optional[float]:
        """
        Вычисляет один динамический параметр.

        Приоритет:
          1. topology_conditions[topology_type] — точное выражение для топологии
          2. formula — общая формула
          3. fallback — статическое значение по умолчанию
        """
        topology    = ctx.get("topology_type", "unknown")
        conditions  = spec.get("topology_conditions", {})
        required    = set(spec.get("required_inputs", []))

        # Выбираем формулу
        formula = conditions.get(topology) or spec.get("formula")

        if formula is None:
            return None

        # Числовой литерал — без eval
        if isinstance(formula, (int, float)):
            return float(formula)

        # Строковая формула — безопасный eval.
        # Whitelist строится из ВСЕХ ключей ctx (не только required_inputs),
        # чтобы обогащённые переменные (_enrich_context: mean_k_sq и др.)
        # были доступны в формулах topology_conditions.
        if isinstance(formula, str):
            all_ctx_tokens = {k for k in ctx if _VALID_TOKEN_RE.match(k)}
            return self._safe_eval(
                formula, required, ctx, spec.get("fallback"),
                allowed_tokens_override=all_ctx_tokens,
            )

        return None

    def _safe_eval(
        self,
        formula: str,
        required_tokens: Set[str],
        ctx: Dict,
        fallback: Optional[Any] = None,
        allowed_tokens_override: Optional[Set[str]] = None,
    ) -> Optional[float]:
        """
        Безопасное вычисление арифметической формулы.

        required_tokens          — обязательные переменные (fallback если отсутствуют).
        allowed_tokens_override  — расширенный whitelist для _validate_formula:
                                   включает обогащённые переменные (_enrich_context).
                                   Если None — whitelist = required_tokens.

        Защита (двойная):
          1. _validate_formula() — whitelist символов + блокировка опасных паттернов.
          2. eval() с __builtins__={} и переменными только из ctx ∩ whitelist.
        """
        whitelist = allowed_tokens_override if allowed_tokens_override is not None                     else required_tokens

        # Валидируем имена токенов
        for tok in whitelist:
            if not _VALID_TOKEN_RE.match(tok):
                logger.error(
                    f"ThresholdCalculator: invalid token in whitelist: {tok!r}"
                )
                return float(fallback) if fallback is not None else None

        # Проверяем обязательные переменные ДО validate_formula —
        # не генерируем ложные предупреждения когда ctx пустой.
        missing = required_tokens - set(ctx.keys())
        if missing:
            logger.debug(
                f"ThresholdCalculator: required vars {missing} absent "
                f"for {formula!r} → fallback={fallback}"
            )
            return float(fallback) if fallback is not None else None

        if not _validate_formula(formula, whitelist):
            logger.error(
                f"ThresholdCalculator: formula failed safety validation: {formula!r}"
            )
            return float(fallback) if fallback is not None else None

        # Словарь для eval: все ключи ctx, входящие в whitelist
        available = {k: ctx[k] for k in whitelist if k in ctx}

        try:
            result = eval(
                formula,
                {"__builtins__": {}},
                available,
            )
            val = float(result)

            if not (0.0 <= val <= 1e6):
                logger.warning(
                    f"ThresholdCalculator: result {val} out of range "
                    f"for formula {formula!r}"
                )
                return float(fallback) if fallback is not None else None

            return round(val, 6)

        except ZeroDivisionError:
            logger.warning(f"ThresholdCalculator: ZeroDivision in formula {formula!r}")
            return float(fallback) if fallback is not None else None
        except Exception as e:
            logger.error(f"ThresholdCalculator: eval error in {formula!r}: {e}")
            return float(fallback) if fallback is not None else None

    def _enrich_context(self, ctx: Dict, model: Dict) -> Dict:
        """
        Дополняет structural_context приближениями для недостающих переменных.

        mean_k_sq вычисляется через аппроксимации по топологии:
          erdos_renyi:  <k²> ≈ k(k+1)     [пуассоновское распределение]
          scale_free:   <k²> ≈ k^1.5 * 2  [степенной хвост, γ≈2.5]
          small_world:  <k²> ≈ k(k+1)*0.8 [меньше дисперсия]
        """
        enriched = dict(ctx)
        topology = ctx.get("topology_type", "erdos_renyi")
        k = float(ctx.get("mean_k", 6.0))

        if "mean_k_sq" not in enriched:
            if topology == "scale_free":
                enriched["mean_k_sq"] = round(k ** 1.5 * 2, 4)
            elif topology == "small_world":
                enriched["mean_k_sq"] = round(k * (k + 1) * 0.8, 4)
            else:
                enriched["mean_k_sq"] = round(k * (k + 1), 4)
            logger.debug(
                f"ThresholdCalculator: estimated mean_k_sq={enriched['mean_k_sq']:.3f} "
                f"for topology={topology}, k={k}"
            )

        return enriched

    @staticmethod
    def infer_context_from_four_d(four_d: Dict) -> Dict:
        """
        Извлекает structural_context из four_d_matrix артефакта.
        Вызывается в MGAPMatcher._build_match() перед compute().

        Топология определяется по C и D (эвристика, см. TopologyValidator).
        """
        s = four_d.get("structure", {})
        C = float(s.get("C", 0.5))
        k = float(s.get("k", 6.0))
        D = float(s.get("D", 2.0))

        # Простая эвристика для определения топологии
        if D > 2.5 and C > 0.6:
            topology = "scale_free"
        elif C >= 0.50 and k < 20:
            topology = "small_world"
        elif C < 0.25:
            topology = "erdos_renyi"
        else:
            topology = "erdos_renyi"

        return {
            "mean_k":          k,
            "clustering_coef": C,
            "topology_type":   topology,
        }
