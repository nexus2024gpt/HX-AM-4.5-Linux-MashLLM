# retry_manager.py — HX-AM v4.5.6
"""
Синхронный RetryManager для повторных попыток LLM-вызовов
при неуспешной валидации (невалидный JSON, пустая гипотеза и т.д.).

НЕ заменяет существующий fallback в llm_client_v_4.py —
работает как дополнительный слой ПОСЛЕ получения ответа.

Точки применения:
  1. question_generator._fill_field()   — короткие гипотезы/механизм
  2. hxam_v_4_server.process_query()   — gen/ver шаги пайплайна
  3. mgap_matcher._improve_blind_spot() — LLM-улучшение слепой зоны
"""

import logging
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger("HXAM.retry")

# ── Конфиг по умолчанию ──────────────────────────────────────────────────────
RETRY_CONFIG = {
    "max_retries_per_provider": 2,   # +2 попытки после первой (итого 3)
    "base_delay_sec": 1.0,
    "backoff_factor": 1.5,           # задержка: 1s → 1.5s → 2.25s
    "min_hypothesis_len": 30,        # минимум символов для hypothesis
    "min_mechanism_len":  20,        # минимум символов для mechanism
}


class RetryResult:
    """Результат retry-цикла с аудит-трейлом."""
    def __init__(self, value: Any, attempts: int, provider: str, repairs: list):
        self.value    = value
        self.attempts = attempts
        self.provider = provider
        self.repairs  = repairs   # список строк с описанием что было исправлено

    def __bool__(self):
        return self.value is not None


# ══════════════════════════════════════════════════════════════════════════════

class RetryManager:
    """
    Синхронный менеджер повторных попыток.

    Пример использования:
        rm = RetryManager()
        result = rm.call_with_retry(
            func=client.generate,
            prompt=prompt,
            validator=lambda text, _: len(text) > 30,
            normalize=lambda raw, _: (raw.strip(), "unknown"),
            context="gen/hypothesis",
        )
        if result:
            text, model = result.value
    """

    def __init__(
        self,
        max_retries: int        = RETRY_CONFIG["max_retries_per_provider"],
        base_delay:  float      = RETRY_CONFIG["base_delay_sec"],
        backoff:     float      = RETRY_CONFIG["backoff_factor"],
    ):
        self.max_retries = max_retries
        self.base_delay  = base_delay
        self.backoff     = backoff

    # ── Основной метод ────────────────────────────────────────────────────────

    def call_with_retry(
        self,
        func:      Callable,
        *args,
        validator: Callable[[Any, Any], bool],
        normalize: Optional[Callable[[Any, Any], Any]] = None,
        context:   str = "llm_call",
        **kwargs,
    ) -> RetryResult:
        """
        Вызывает func(*args, **kwargs), проверяет через validator,
        повторяет при неуспехе.

        func должна возвращать (raw_text, model_name) — как LLMClient.generate().

        validator(normalized_value, model_name) -> bool

        Возвращает RetryResult. При исчерпании попыток — RetryResult(value=None).
        """
        repairs     = []
        last_raw    = None
        last_model  = "unknown"

        for attempt in range(self.max_retries + 1):
            try:
                raw, model = func(*args, **kwargs)
                last_raw   = raw
                last_model = model

                if not raw or raw.startswith("[Generator error]") or raw.startswith("[Verifier error]"):
                    reason = f"attempt {attempt+1}: empty/error response from {model}"
                    repairs.append(reason)
                    logger.warning(f"[RetryManager:{context}] {reason}")
                else:
                    # Нормализуем если нужно
                    normalized = normalize(raw, model) if normalize else (raw, model)

                    # Валидируем
                    if validator(normalized, model):
                        if attempt > 0:
                            logger.info(
                                f"[RetryManager:{context}] success on attempt {attempt+1} "
                                f"via {model}"
                            )
                        return RetryResult(
                            value=normalized,
                            attempts=attempt + 1,
                            provider=model,
                            repairs=repairs,
                        )
                    else:
                        reason = f"attempt {attempt+1}: validation failed (model={model})"
                        repairs.append(reason)
                        logger.warning(f"[RetryManager:{context}] {reason}")

            except Exception as e:
                reason = f"attempt {attempt+1}: exception — {str(e)[:80]}"
                repairs.append(reason)
                logger.warning(f"[RetryManager:{context}] {reason}")

            # Пауза перед следующей попыткой
            if attempt < self.max_retries:
                delay = self.base_delay * (self.backoff ** attempt)
                logger.info(f"[RetryManager:{context}] retry in {delay:.1f}s...")
                time.sleep(delay)

        logger.error(
            f"[RetryManager:{context}] all {self.max_retries+1} attempts failed. "
            f"Last model: {last_model}"
        )
        return RetryResult(value=None, attempts=self.max_retries+1,
                           provider=last_model, repairs=repairs)

    # ── Хелперы-валидаторы для повторного использования ───────────────────────

    @staticmethod
    def validator_gen(normalized: Tuple, model: str) -> bool:
        """Валидатор для gen-ответа: (normalized_dict, repairs, is_ok)."""
        if not normalized or len(normalized) < 3:
            return False
        gen_dict, repairs, is_ok = normalized
        if not is_ok:
            return False
        hyp = str(gen_dict.get("hypothesis", "")).strip()
        mec = str(gen_dict.get("mechanism",  "")).strip()
        return (
            len(hyp) >= RETRY_CONFIG["min_hypothesis_len"]
            and len(mec) >= RETRY_CONFIG["min_mechanism_len"]
        )

    @staticmethod
    def validator_ver(normalized: Tuple, model: str) -> bool:
        """Валидатор для ver-ответа: (normalized_dict, repairs, is_ok)."""
        if not normalized or len(normalized) < 3:
            return False
        ver_dict, repairs, is_ok = normalized
        if not is_ok:
            return False
        verdict = str(ver_dict.get("verdict", "")).strip().upper()
        return verdict in ("VALID", "WEAK", "FALSE")

    @staticmethod
    def validator_text(normalized: Tuple, model: str) -> bool:
        """Валидатор для простого текстового ответа (min 15 символов)."""
        if not normalized:
            return False
        text = normalized[0] if isinstance(normalized, tuple) else normalized
        return bool(text) and len(str(text).strip()) >= 15

    @staticmethod
    def validator_field(min_len: int = 20):
        """Фабрика валидатора для одного текстового поля."""
        def _v(normalized: Tuple, model: str) -> bool:
            text = normalized[0] if isinstance(normalized, tuple) else normalized
            return bool(text) and len(str(text).strip()) >= min_len
        return _v


# ── Глобальный синглтон ───────────────────────────────────────────────────────
retry_manager = RetryManager()