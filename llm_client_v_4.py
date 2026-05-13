# llm_client_v_4.py — HX-AM v4.5.3 [FIXED]
"""
v4.5.3 — исправления:
  - generate() и verify(): убран стray-вызов _try_all_mesh(prompt, role=...)
  - _try_mash(): вызывает _try_all_mesh(candidates, prompt, role) корректно
  - _try_all_mesh(candidates, prompt, role): правильная сигнатура
  - matches_role_hint() вместо matches_role()
  - /api/status string-ответ обрабатывается в адаптере
"""

import logging
import os
import time
from threading import Lock
from typing import Dict, List, Optional, Tuple

import requests
from api_usage_tracker import tracker, ProviderConfig
from mash_llm_adapter import mash_adapter, MashModel

logger = logging.getLogger("HXAM.llm")

# ── Таймауты ─────────────────────────────────────────────────────
TIMEOUT_LOCAL       = 30
TIMEOUT_REMOTE_GEN  = 60
TIMEOUT_REMOTE_VER  = 90
MASH_CALL_TIMEOUT_GEN = int(os.getenv("MASH_CALL_TIMEOUT_GEN", "45"))
MASH_CALL_TIMEOUT_VER = int(os.getenv("MASH_CALL_TIMEOUT_VER", "75"))

# ── Circuit breaker (только для внешних провайдеров) ─────────────
CIRCUIT_COOLDOWN   = 180
_circuit_lock      = Lock()
_circuit_state: Dict[str, float] = {}

_MASH_BASE = os.getenv("MASH_BASE_URL", "http://localhost:9337")


def _is_local(api_base: str) -> bool:
    return "localhost" in api_base or "127.0.0.1" in api_base


def _is_mash_provider(p: ProviderConfig) -> bool:
    return _MASH_BASE.rstrip("/") in p.api_base.rstrip("/")


def _get_timeout(api_base: str, role: str) -> int:
    if _is_local(api_base):
        return TIMEOUT_LOCAL
    return TIMEOUT_REMOTE_VER if role == "verifier" else TIMEOUT_REMOTE_GEN


def _circuit_open(provider_id: str) -> bool:
    with _circuit_lock:
        until = _circuit_state.get(provider_id, 0)
        if until > time.time():
            remaining = int(until - time.time())
            logger.warning(
                f"[Circuit Breaker] {provider_id} заблокирован ещё {remaining}s"
            )
            return True
        if provider_id in _circuit_state and until <= time.time():
            del _circuit_state[provider_id]
    return False


def _circuit_trip(provider_id: str):
    with _circuit_lock:
        _circuit_state[provider_id] = time.time() + CIRCUIT_COOLDOWN
    logger.warning(f"[Circuit Breaker] {provider_id} ОТКЛЮЧЁН на {CIRCUIT_COOLDOWN}s")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ════════════════════════════════════════════════════════════════
class LLMClient:

    def __init__(self):
        self.hf_chat_url = os.getenv(
            "HF_CHAT_COMPLETIONS_URL",
            "https://router.huggingface.co/v1/chat/completions",
        )

    # ── Публичные методы ──────────────────────────────────────────

    def generate(self, prompt: str) -> Tuple[str, str]:
        """
        1. Mash (лучшая ready-модель по RAM/score) + fallback по всем Mash
        2. Резервные провайдеры из tracker
        """
        # Попытка через Mash
        result = self._try_mash(prompt, role="generator")
        if result is not None:
            return result

        # Резервные провайдеры (не-Mash)
        providers = [
            p for p in tracker.get_providers_for_role("generator")
            if not _is_mash_provider(p)
        ]
        if not providers:
            providers = tracker.get_providers_for_role("generator")

        if not providers:
            return "[Generator error] no providers configured", "none"

        return self._call_providers(providers, prompt, role="generator")

    def verify(self, statement: str, context: str = "") -> Tuple[str, str]:
        """
        1. Mash (лучшая ready-модель по RAM/score) + fallback по всем Mash
        2. Резервные провайдеры из tracker
        """
        full_prompt = f"Context: {context}\n\n{statement}" if context else statement

        result = self._try_mash(full_prompt, role="verifier")
        if result is not None:
            return result

        providers = [
            p for p in tracker.get_providers_for_role("verifier")
            if not _is_mash_provider(p)
        ]
        if not providers:
            providers = tracker.get_providers_for_role("verifier")

        if not providers:
            return "[Verifier error] no providers configured", "none"

        return self._call_providers(providers, full_prompt, role="verifier")

    # ── Mash-специфичные методы ───────────────────────────────────

    def _try_mash(
        self, prompt: str, role: str
    ) -> Optional[Tuple[str, str]]:
        """
        Пробует лучшую Mash-модель, затем остальные по убыванию score.
        Не активирует Circuit Breaker.
        """
        if not mash_adapter.is_healthy():
            logger.debug(f"_try_mash({role}): адаптер не healthy → пропуск")
            return None

        all_ready: List[MashModel] = mash_adapter.get_all_ready_sorted()
        if not all_ready:
            logger.debug(f"_try_mash({role}): нет ready-моделей → fallback")
            return None

        # Лучшая по score
        best = all_ready[0]
        text, t_in, t_out, err = self._call_mash(best, prompt, role)

        if text:
            self._record_mash_call(role, t_in, t_out)
            logger.info(
                f"LLMClient.{role} ✓ Mash[{best.canonical_name}] "
                f"ram={best.ram_gb}GB | in={t_in} out={t_out}"
            )
            return text, f"mash/{best.canonical_name}"

        logger.warning(
            f"LLMClient._try_mash({role}): "
            f"'{best.canonical_name}' не ответила — {err[:100]}"
        )

        # Перебираем оставшиеся Mash-модели
        return self._try_all_mesh(all_ready[1:], prompt, role)

    def _try_all_mesh(
        self,
        candidates: List[MashModel],
        prompt: str,
        role: str,
    ) -> Optional[Tuple[str, str]]:
        """
        Перебирает кандидатов по порядку (уже отсортированы по score).
        Circuit Breaker не активируется.
        """
        for model in candidates:
            text, t_in, t_out, err = self._call_mash(model, prompt, role)
            if text:
                self._record_mash_call(role, t_in, t_out)
                logger.info(
                    f"LLMClient.{role} ✓ Mash[{model.canonical_name}] (fallback) "
                    f"ram={model.ram_gb}GB | in={t_in} out={t_out}"
                )
                return text, f"mash/{model.canonical_name}"

            logger.warning(
                f"LLMClient._try_all_mesh({role}): "
                f"'{model.canonical_name}' не ответила — {err[:100]}"
            )

        return None

    def _call_mash(
        self,
        model: MashModel,
        prompt: str,
        role: str,
    ) -> Tuple[str, int, int, str]:
        """
        Targeted запрос к модели MashLLM.
        Перебирает all_name_variants() при HTTP 400.
        """
        url     = f"{_MASH_BASE}/v1/chat/completions"
        timeout = MASH_CALL_TIMEOUT_VER if role == "verifier" else MASH_CALL_TIMEOUT_GEN
        temp    = 0.3 if role == "verifier" else 0.7
        max_tok = 2048 if role == "verifier" else 1024
        api_key = os.getenv("MASH_API_KEY", "mash-local")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        variants  = model.all_name_variants()
        last_err  = "нет вариантов имени"

        for attempt, model_name in enumerate(variants):
            payload = {
                "model":       model_name,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens":  max_tok,
            }

            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=timeout
                )

                if resp.status_code == 503:
                    last_err = f"HTTP 503: модель '{model_name}' недоступна"
                    logger.warning(
                        f"Попытка вызвать модель {model_name} "
                        f"не удалась: Mash timeout ({timeout}s) for {model_name}"
                    )
                    break  # 503 — не пробуем другое имя

                if resp.status_code == 400:
                    last_err = (
                        f"Mash HTTP 400: 400 Client Error: "
                        f"Bad Request for url: {url}"
                    )
                    logger.warning(
                        f"Попытка вызвать модель {model_name} "
                        f"не удалась: {last_err}"
                    )
                    if attempt < len(variants) - 1:
                        logger.debug(
                            f"_call_mash: 400 на '{model_name}', "
                            f"пробуем '{variants[attempt+1]}'"
                        )
                    continue  # 400 — пробуем следующее имя

                resp.raise_for_status()

                rj      = resp.json()
                content = rj["choices"][0]["message"]["content"]
                usage   = rj.get("usage", {})
                t_in    = usage.get("prompt_tokens",     _estimate_tokens(prompt))
                t_out   = usage.get("completion_tokens", _estimate_tokens(content or ""))

                if content and content.strip():
                    actual = (
                        rj.get("model")
                        or rj["choices"][0].get("model")
                        or model_name
                    )
                    logger.info(f"Mash call succeeded with model name: {actual}")
                    return content, t_in, t_out, ""

                last_err = "пустой content в ответе Mash"

            except requests.exceptions.Timeout:
                last_err = f"Mash timeout ({timeout}s) for {model_name}"
                logger.warning(
                    f"Попытка вызвать модель {model_name} "
                    f"не удалась: {last_err}"
                )
                break  # таймаут — нет смысла пробовать другое имя

            except requests.exceptions.ConnectionError as e:
                last_err = f"Mash ConnectionError: {str(e)[:80]}"
                break

            except requests.HTTPError as e:
                status   = (
                    getattr(e.response, "status_code", 0)
                    if e.response is not None else 0
                )
                last_err = f"Mash HTTP {status}: {str(e)[:80]}"
                logger.warning(
                    f"Попытка вызвать модель {model_name} "
                    f"не удалась: {last_err}"
                )
                if status not in (400, 404):
                    break

            except (KeyError, IndexError) as e:
                last_err = f"Mash неожиданная структура: {e}"
                break

            except Exception as e:
                last_err = str(e)[:150]
                break

        return "", 0, 0, last_err

    def _record_mash_call(self, role: str, t_in: int, t_out: int):
        """Записывает статистику вызова в tracker (Mash-провайдер)."""
        for p in tracker.get_providers_for_role(role):
            if _is_mash_provider(p):
                tracker.record_call(p.id, tokens_in=t_in, tokens_out=t_out)
                return

    def _find_mash_provider(self, role: str) -> Optional[ProviderConfig]:
        for p in tracker.get_providers_for_role(role):
            if _is_mash_provider(p):
                return p
        return None

    # ── Резервные провайдеры (tracker) ───────────────────────────

    def _call_providers(
        self,
        providers: List[ProviderConfig],
        prompt: str,
        role: str,
    ) -> Tuple[str, str]:
        prefix = "Generator" if role == "generator" else "Verifier"

        for p in providers:
            if _circuit_open(p.id):
                continue

            text, t_in, t_out, err = self._call(p, prompt, role=role)

            if text:
                tracker.record_call(p.id, tokens_in=t_in, tokens_out=t_out)
                logger.info(
                    f"LLMClient.{role} ✓ {p.label} [резерв] "
                    f"| in={t_in} out={t_out}"
                )
                return text, f"{p.provider}/{p.model}"

            is_fault = any(
                kw in err for kw in (
                    "timed out", "timeout", "503",
                    "Service Unavailable", "ConnectionError",
                    "Connection refused",
                )
            )
            tracker.record_call(p.id, error=True, error_msg=err)
            logger.warning(f"LLMClient.{role} ✗ {p.label} [резерв]: {err[:150]}")

            if is_fault and _is_local(p.api_base) and not _is_mash_provider(p):
                _circuit_trip(p.id)

        return f"[{prefix} error] all providers failed", "none"

    # ── Низкоуровневые вызовы ────────────────────────────────────

    def _call(
        self, p: ProviderConfig, prompt: str, role: str = "generator"
    ) -> Tuple[str, int, int, str]:
        if not p.api_key:
            return "", 0, 0, "api_key not set"
        try:
            if p.provider == "gemini":
                return self._call_gemini(p, prompt, role=role)
            return self._call_openai_compat(p, prompt, role=role)
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_openai_compat(
        self, p: ProviderConfig, prompt: str, role: str = "generator"
    ) -> Tuple[str, int, int, str]:
        url = (
            self.hf_chat_url
            if p.provider == "huggingface"
            else f"{p.api_base}/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {p.api_key}",
            "Content-Type":  "application/json",
        }
        temp    = 0.5 if p.provider in ("huggingface", "nvidia") else (
                  0.3 if role == "verifier" else 0.7)
        max_tok = 2048 if role == "verifier" else 1024
        timeout = getattr(p, "timeout", None) or _get_timeout(p.api_base, role)

        payload = {
            "model":       p.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens":  max_tok,
        }

        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )
            if resp.status_code == 503:
                return "", 0, 0, "HTTP 503: Service Unavailable"
            resp.raise_for_status()
            rj      = resp.json()
            content = rj["choices"][0]["message"]["content"]
            usage   = rj.get("usage", {})
            t_in    = usage.get("prompt_tokens",     _estimate_tokens(prompt))
            t_out   = usage.get("completion_tokens", _estimate_tokens(content or ""))
            if content and content.strip():
                return content, t_in, t_out, ""
            return "", t_in, 0, "empty content"

        except requests.exceptions.Timeout:
            return "", 0, 0, f"Read timed out. (timeout={timeout})"
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", 0) if e.response else 0
            return "", 0, 0, f"HTTP {status}: {str(e)[:150]}"
        except requests.exceptions.ConnectionError as e:
            return "", 0, 0, f"ConnectionError: {str(e)[:120]}"
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_gemini(
        self, p: ProviderConfig, prompt: str, role: str = "generator"
    ) -> Tuple[str, int, int, str]:
        url = (
            f"{p.api_base}/models/{p.model}:generateContent?key={p.api_key}"
        )
        max_out = 4096 if role == "verifier" else 1024
        timeout = getattr(p, "timeout", None) or TIMEOUT_REMOTE_VER

        payload = {
            "contents":         [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_out},
        }

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            rj   = resp.json()
            text = rj["candidates"][0]["content"]["parts"][0]["text"]
            usage = rj.get("usageMetadata", {})
            t_in  = usage.get("promptTokenCount",     _estimate_tokens(prompt))
            t_out = usage.get("candidatesTokenCount", _estimate_tokens(text or ""))
            if text and text.strip():
                return text, t_in, t_out, ""
            return "", t_in, 0, "empty content"

        except requests.exceptions.Timeout:
            return "", 0, 0, f"Read timed out. (timeout={timeout})"
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", 0) if e.response else 0
            return "", 0, 0, f"HTTP {status}: {str(e)[:150]}"
        except Exception as e:
            return "", 0, 0, str(e)[:200]


def circuit_status() -> Dict[str, float]:
    now = time.time()
    with _circuit_lock:
        return {
            pid: round(until - now, 1)
            for pid, until in _circuit_state.items()
            if until > now
        }
