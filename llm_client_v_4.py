# llm_client_v_4.py — HX-AM v4.5.1 [SmartRouter edition]
"""
v4.5.1 исправления [SmartRouter]:
  - Circuit breaker: провайдеры с таймаутом блокируются на CIRCUIT_COOLDOWN секунд
  - Динамический таймаут: localhost/127.0.0.1 → TIMEOUT_LOCAL (30s),
    внешние API → TIMEOUT_REMOTE_GEN (60s) / TIMEOUT_REMOTE_VER (90s)
  - _call_openai_compat: timeout берётся из ProviderConfig.timeout если задан
  - Логирование: предупреждение при срабатывании circuit breaker
  - При 503 от локального провайдера — мгновенный skip (не ждём таймаут)
"""

import logging
import os
import time
from threading import Lock
from typing import Dict, Tuple

import requests
from api_usage_tracker import tracker, ProviderConfig

logger = logging.getLogger("HXAM.llm")

# ── Таймауты ────────────────────────────────────────────────────────
TIMEOUT_LOCAL      = 30    # localhost / 127.0.0.1 — быстрый фейл
TIMEOUT_REMOTE_GEN = 60    # внешний генератор
TIMEOUT_REMOTE_VER = 90    # внешний верификатор (Gemini нужно больше)

# ── Circuit breaker ─────────────────────────────────────────────────
CIRCUIT_COOLDOWN   = 180   # секунд блокировки после таймаута (3 мин)
CIRCUIT_FAIL_CODES = {
    "ReadTimeout", "ConnectTimeout",
    "ConnectionError",          # 503 Service Unavailable
}

_circuit_lock   = Lock()
_circuit_state: Dict[str, float] = {}   # provider_id → timestamp_unblock


def _is_local(api_base: str) -> bool:
    return "localhost" in api_base or "127.0.0.1" in api_base


def _get_timeout(api_base: str, role: str) -> int:
    if _is_local(api_base):
        return TIMEOUT_LOCAL
    return TIMEOUT_REMOTE_VER if role == "verifier" else TIMEOUT_REMOTE_GEN


def _circuit_open(provider_id: str) -> bool:
    """True если провайдер сейчас заблокирован circuit breaker."""
    with _circuit_lock:
        until = _circuit_state.get(provider_id, 0)
        if until > time.time():
            remaining = int(until - time.time())
            logger.warning(
                f"[Circuit Breaker] {provider_id} заблокирован ещё {remaining}s — skip"
            )
            return True
        # Снимаем блокировку если время истекло
        if provider_id in _circuit_state and until <= time.time():
            del _circuit_state[provider_id]
    return False


def _circuit_trip(provider_id: str):
    """Блокировать провайдер на CIRCUIT_COOLDOWN секунд."""
    with _circuit_lock:
        _circuit_state[provider_id] = time.time() + CIRCUIT_COOLDOWN
    logger.warning(
        f"[Circuit Breaker] {provider_id} ОТКЛЮЧЁН на {CIRCUIT_COOLDOWN}s "
        f"(таймаут/503)"
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:

    def __init__(self):
        self.hf_chat_url = os.getenv(
            "HF_CHAT_COMPLETIONS_URL",
            "https://router.huggingface.co/v1/chat/completions",
        )

    # ── Публичные методы ────────────────────────────────────────────

    def generate(self, prompt: str) -> Tuple[str, str]:
        providers = tracker.get_providers_for_role("generator")
        if not providers:
            return "[Generator error] no providers configured", "none"

        for p in providers:
            if _circuit_open(p.id):
                continue
            text, tokens_in, tokens_out, err_msg = self._call(p, prompt, role="generator")
            if text:
                tracker.record_call(p.id, tokens_in=tokens_in, tokens_out=tokens_out)
                logger.info(
                    f"LLMClient.generate ✓ {p.label} | in={tokens_in} out={tokens_out}"
                )
                return text, f"{p.provider}/{p.model}"
            else:
                is_circuit_fault = any(
                    kw in err_msg for kw in ("timed out", "timeout", "503", "Service Unavailable",
                                             "ConnectionError", "Connection refused")
                )
                tracker.record_call(p.id, error=True, error_msg=err_msg)
                logger.warning(f"LLMClient.generate ✗ {p.label}: {err_msg[:150]}")
                if is_circuit_fault and _is_local(p.api_base):
                    _circuit_trip(p.id)

        return "[Generator error] all providers failed", "none"

    def verify(self, statement: str, context: str = "") -> Tuple[str, str]:
        full_prompt = f"Context: {context}\n\n{statement}" if context else statement
        providers = tracker.get_providers_for_role("verifier")
        if not providers:
            return "[Verifier error] no providers configured", "none"

        for p in providers:
            if _circuit_open(p.id):
                continue
            text, tokens_in, tokens_out, err_msg = self._call(p, full_prompt, role="verifier")
            if text:
                tracker.record_call(p.id, tokens_in=tokens_in, tokens_out=tokens_out)
                logger.info(
                    f"LLMClient.verify ✓ {p.label} | in={tokens_in} out={tokens_out}"
                )
                return text, f"{p.provider}/{p.model}"
            else:
                is_circuit_fault = any(
                    kw in err_msg for kw in ("timed out", "timeout", "503", "Service Unavailable",
                                             "ConnectionError", "Connection refused")
                )
                tracker.record_call(p.id, error=True, error_msg=err_msg)
                logger.warning(f"LLMClient.verify ✗ {p.label}: {err_msg[:150]}")
                if is_circuit_fault and _is_local(p.api_base):
                    _circuit_trip(p.id)

        return "[Verifier error] all providers failed", "none"

    # ── Внутренние методы ───────────────────────────────────────────

    def _call(
        self, p: ProviderConfig, prompt: str, role: str = "generator"
    ) -> Tuple[str, int, int, str]:
        if not p.api_key:
            return "", 0, 0, "api_key not set"
        try:
            if p.provider == "gemini":
                return self._call_gemini(p, prompt, role=role)
            else:
                return self._call_openai_compat(p, prompt, role=role)
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_openai_compat(
        self, p: ProviderConfig, prompt: str, role: str = "generator"
    ) -> Tuple[str, int, int, str]:
        if p.provider == "huggingface":
            url = self.hf_chat_url
        else:
            url = f"{p.api_base}/chat/completions"

        headers = {
            "Authorization": f"Bearer {p.api_key}",
            "Content-Type": "application/json",
        }

        temperature = (
            0.5 if p.provider in ("huggingface", "nvidia")
            else (0.3 if role == "verifier" else 0.7)
        )
        max_tokens = 2048 if role == "verifier" else 1024

        # ── Динамический таймаут ─────────────────────────────────────
        # Если в ProviderConfig есть поле timeout — используем его,
        # иначе выбираем по типу API base.
        timeout = getattr(p, "timeout", None) or _get_timeout(p.api_base, role)

        payload = {
            "model": p.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            # Мгновенный skip на 503 — не ждём повторно
            if resp.status_code == 503:
                return "", 0, 0, f"HTTP 503: Service Unavailable"
            resp.raise_for_status()
            rj = resp.json()
            content = rj["choices"][0]["message"]["content"]
            usage = rj.get("usage", {})
            tokens_in  = usage.get("prompt_tokens",     _estimate_tokens(prompt))
            tokens_out = usage.get("completion_tokens", _estimate_tokens(content or ""))

            if content and content.strip():
                return content, tokens_in, tokens_out, ""
            return "", tokens_in, 0, "empty content in response"

        except requests.exceptions.Timeout:
            return "", 0, 0, f"Read timed out. (read timeout={timeout})"
        except requests.HTTPError as e:
            status = (
                getattr(e.response, "status_code", 0)
                if e.response is not None else 0
            )
            return "", 0, 0, f"HTTP {status}: {str(e)[:150]}"
        except requests.exceptions.ConnectionError as e:
            return "", 0, 0, f"ConnectionError: {str(e)[:120]}"
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_gemini(
        self, p: ProviderConfig, prompt: str, role: str = "generator"
    ) -> Tuple[str, int, int, str]:
        url = f"{p.api_base}/models/{p.model}:generateContent?key={p.api_key}"
        max_output = 4096 if role == "verifier" else 1024
        timeout = getattr(p, "timeout", None) or TIMEOUT_REMOTE_VER

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output},
        }

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            rj = resp.json()
            text = rj["candidates"][0]["content"]["parts"][0]["text"]
            usage = rj.get("usageMetadata", {})
            tokens_in  = usage.get("promptTokenCount",      _estimate_tokens(prompt))
            tokens_out = usage.get("candidatesTokenCount",  _estimate_tokens(text or ""))

            if text and text.strip():
                return text, tokens_in, tokens_out, ""
            return "", tokens_in, 0, "empty content in response"

        except requests.exceptions.Timeout:
            return "", 0, 0, f"Read timed out. (read timeout={timeout})"
        except requests.HTTPError as e:
            status = (
                getattr(e.response, "status_code", 0)
                if e.response is not None else 0
            )
            return "", 0, 0, f"HTTP {status}: {str(e)[:150]}"
        except Exception as e:
            return "", 0, 0, str(e)[:200]


# ── Утилита для просмотра состояния circuit breaker ─────────────────

def circuit_status() -> Dict[str, float]:
    """Возвращает текущее состояние circuit breaker (для отладки)."""
    now = time.time()
    with _circuit_lock:
        return {
            pid: round(until - now, 1)
            for pid, until in _circuit_state.items()
            if until > now
        }
