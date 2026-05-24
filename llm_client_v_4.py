# llm_client_v_4.py — HX-AM v4.5.5 [Mesh-first priority + streaming]

import logging
import json
import os
import re
import time
from threading import Lock
from typing import Dict, Tuple

import requests
from api_usage_tracker import tracker

logger = logging.getLogger("HXAM.llm")

# ── Таймауты не-streaming вызовов ─────────────────────────────────────────────
TIMEOUT_LOCAL      = 240
TIMEOUT_REMOTE_GEN = 60
TIMEOUT_REMOTE_VER = 90

# ── Streaming ─────────────────────────────────────────────────────────────────
# Время ожидания ПЕРВОГО чанка от Mesh до переключения на local_llama.
# Расчёт: prefill Qwen3-9B на i5-6300U ≈ 15-25 сек + 4 сек HTTP overhead.
MESH_FIRST_CHUNK_TIMEOUT = 35   # сек

# Пауза между чанками после получения первого (idle).
# Если Mesh замолчал в середине генерации — фатальная ошибка модели.
STREAM_IDLE_TIMEOUT      = 60   # сек

# Общий потолок стриминг-сессии (защита от зависания навсегда)
STREAM_TOTAL_TIMEOUT     = 600  # сек

# ── Таймауты Mesh (передаются в requests.post timeout=) ───────────────────────
MESH_TIMEOUT_GEN = int(os.getenv("MASH_CALL_TIMEOUT_GEN", "600"))
MESH_TIMEOUT_VER = int(os.getenv("MASH_CALL_TIMEOUT_VER", "600"))

# ── Circuit Breaker ────────────────────────────────────────────────────────────
CIRCUIT_COOLDOWN = 180
_circuit_lock    = Lock()
_circuit_state: Dict[str, float] = {}

_MESH_BASE = os.getenv("MASH_BASE_URL", "http://localhost:9337")

# Специальный CB-ключ для Mesh (не привязан к provider_id из tracker)
_MESH_CB_KEY = "__mesh_auto__"


# ── Think-block stripper (Qwen3 / DeepSeek reasoning mode) ───────────────────
_THINK_RE      = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*$",         flags=re.DOTALL | re.IGNORECASE)

def _strip_think(text: str) -> str:
    """Удаляет <think>…</think> из вывода LLM. Работает с закрытыми и незакрытыми блоками."""
    if not text or "<think>" not in text.lower():
        return text
    result = _THINK_RE.sub("", text)
    result = _THINK_OPEN_RE.sub("", result)
    result = result.lstrip("\n").strip()
    if text != result:
        logger.debug(f"_strip_think: удалено {len(text) - len(result)} символов")
    return result


def _is_local(api_base: str) -> bool:
    return "localhost" in api_base or "127.0.0.1" in api_base


def _get_timeout(api_base: str, role: str) -> int:
    if _is_local(api_base):
        return TIMEOUT_LOCAL
    return TIMEOUT_REMOTE_VER if role == "verifier" else TIMEOUT_REMOTE_GEN


def _circuit_open(provider_id: str) -> bool:
    with _circuit_lock:
        until = _circuit_state.get(provider_id, 0)
        if until > time.time():
            return True
        if provider_id in _circuit_state:
            del _circuit_state[provider_id]
    return False


def _circuit_trip(provider_id: str, cooldown: int = CIRCUIT_COOLDOWN):
    with _circuit_lock:
        _circuit_state[provider_id] = time.time() + cooldown
    logger.warning(f"[CB] {provider_id} заблокирован на {cooldown}s")


def _circuit_reset(provider_id: str):
    with _circuit_lock:
        _circuit_state.pop(provider_id, None)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_sse_line(line: bytes) -> str:
    """
    Парсит одну SSE-строку формата: data: {...}
    Возвращает:
      '\x00'  — маркер [DONE], стрим завершён
      ''      — пустая строка / keepalive / не-data строка
      str     — токен контента
    """
    if not line.startswith(b'data: '):
        return ''
    payload = line[6:].decode('utf-8', errors='replace').strip()
    if payload == '[DONE]':
        return '\x00'
    try:
        chunk   = json.loads(payload)
        choices = chunk.get('choices', [])
        if not choices:
            return ''
        delta = choices[0].get('delta', {})
        return delta.get('content', '') or ''
    except Exception:
        return ''


class LLMClient:

    def generate(self, prompt: str) -> Tuple[str, str]:
        return self._call_with_priority(prompt, role="generator")

    def verify(self, statement: str, context: str = "") -> Tuple[str, str]:
        full_prompt = f"Context: {context}\n\n{statement}" if context else statement
        return self._call_with_priority(full_prompt, role="verifier")

    def _call_with_priority(self, prompt: str, role: str) -> Tuple[str, str]:
        """
        Приоритет:
          1. Mesh        — мощные модели, но нестабилен.
                           При first_chunk_timeout → немедленно local.
          2. local_llama — стабильный fallback.
          3. Cloud       — последний рубеж.
        """
        error_prefix = "Generator" if role == "generator" else "Verifier"

        # ── 1. Mesh ───────────────────────────────────────────────────────────
        if not _circuit_open(_MESH_CB_KEY):
            text, model, err = self._call_mesh_auto(prompt, role)
            if text:
                logger.info(
                    f"LLMClient.{role} ✓ Mesh | "
                    f"tokens≈{_estimate_tokens(text)}"
                )
                return text, f"mesh/{model}"
            # Логируем причину, но НЕ останавливаемся
            logger.warning(f"LLMClient.{role}: Mesh failed ({err[:80]})")
        else:
            logger.info(f"LLMClient.{role}: Mesh CB открыт, пропускаем")

        # ── 2. local_llama ────────────────────────────────────────────────────
        local_providers = [
            p for p in tracker.get_providers_for_role(role)
            if p.provider == "local_llama" and p.enabled
        ]
        for p in local_providers:
            if _circuit_open(p.id):
                continue
            text, t_in, t_out, err = self._call(p, prompt, role)
            if text:
                tracker.record_call(p.id, tokens_in=t_in, tokens_out=t_out)
                logger.info(
                    f"LLMClient.{role} ✓ local {p.label} | "
                    f"in={t_in} out={t_out}"
                )
                return text, f"local/{p.model}"
            tracker.record_call(p.id, error=True, error_msg=err)
            logger.warning(f"LLMClient.{role}: local {p.label} failed ({err[:80]})")
            if any(kw in err for kw in ("timed out", "timeout", "503", "ConnectionError")):
                _circuit_trip(p.id)

        # ── 3. Cloud providers ────────────────────────────────────────────────
        cloud_providers = [
            p for p in tracker.get_providers_for_role(role)
            if p.provider not in ("local_llama", "mesh") and p.enabled
        ]
        if not cloud_providers:
            return f"[{error_prefix} error] all providers failed", "none"

        return self._call_providers(cloud_providers, prompt, role)

    def _call_mesh_auto(self, prompt: str, role: str) -> Tuple[str, str, str]:
        """
        Вызов MeshLLM с двухфазным streaming-таймаутом:
          Фаза 1 — ожидание ПЕРВОГО чанка: MESH_FIRST_CHUNK_TIMEOUT сек.
                   Если молчит → возвращаем ошибку, caller переходит к local_llama.
          Фаза 2 — ожидание СЛЕДУЮЩИХ чанков: STREAM_IDLE_TIMEOUT сек между токенами.
                   Если замолчал в середине → фатальная ошибка.
        Fallback: при любом исключении не связанном с таймаутом первого чанка
                  пробуем запрос без stream=True.
        """
        if _circuit_open(_MESH_CB_KEY):
            remaining = _circuit_state.get(_MESH_CB_KEY, 0) - time.time()
            return "", "", f"Mesh CB открыт ещё {remaining:.0f}s"

        url         = f"{_MESH_BASE}/v1/chat/completions"
        temperature = 0.3 if role == "verifier" else 0.7
        max_tokens  = 4096 if role == "generator" else 2048
        headers     = {"Content-Type": "application/json"}

        payload = {
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      True,
        }

        logger.debug(
            f"Mesh [{role}] stream=True | "
            f"prompt={len(prompt)}ch max_tokens={max_tokens} "
            f"first_chunk_timeout={MESH_FIRST_CHUNK_TIMEOUT}s"
        )

        t_start = time.time()

        # ── Streaming ─────────────────────────────────────────────────────────
        try:
            full_text:      list[str] = []
            first_chunk_ok: bool      = False
            last_chunk_at:  float     = t_start

            with requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=STREAM_TOTAL_TIMEOUT,
            ) as resp:
                if resp.status_code == 503:
                    return "", "", "HTTP 503: Mesh недоступен"
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue

                    now = time.time()

                    # ── Фаза 1: ждём первый токен ─────────────────────────────
                    if not first_chunk_ok:
                        if now - t_start > MESH_FIRST_CHUNK_TIMEOUT:
                            # Mesh молчит → переключаемся на local_llama
                            logger.warning(
                                f"Mesh first-chunk timeout "
                                f"({MESH_FIRST_CHUNK_TIMEOUT}s) → fallback local"
                            )
                            _circuit_trip(_MESH_CB_KEY, cooldown=90)
                            return "", "", (
                                f"first_chunk_timeout: Mesh молчал "
                                f"{MESH_FIRST_CHUNK_TIMEOUT}s"
                            )

                    # ── Фаза 2: idle между токенами ───────────────────────────
                    else:
                        if now - last_chunk_at > STREAM_IDLE_TIMEOUT:
                            raise TimeoutError(
                                f"Mesh idle {STREAM_IDLE_TIMEOUT}s "
                                f"после {len(full_text)} фрагм."
                            )

                    token = _parse_sse_line(raw_line)

                    if token == '\x00':   # [DONE]
                        break
                    if token:
                        full_text.append(token)
                        last_chunk_at = now
                        if not first_chunk_ok:
                            first_chunk_ok = True
                            elapsed = now - t_start
                            logger.info(
                                f"Mesh [{role}] первый токен за {elapsed:.1f}s"
                            )

            final = _strip_think(''.join(full_text))
            if not final.strip():
                raise ValueError("Mesh streaming: пустой итоговый текст")

            total = time.time() - t_start
            logger.info(
                f"Mesh [{role}] ✓ streaming | "
                f"tokens≈{_estimate_tokens(final)} | "
                f"total={total:.1f}s"
            )
            _circuit_reset(_MESH_CB_KEY)   # сбрасываем CB при успехе
            return final, "mesh/auto", ""

        except TimeoutError as e:
            # idle-таймаут в середине генерации
            logger.error(f"Mesh streaming idle-timeout: {e}")
            _circuit_trip(_MESH_CB_KEY, cooldown=120)
            return "", "", str(e)

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Mesh недоступен (ConnectionError): {str(e)[:80]}")
            _circuit_trip(_MESH_CB_KEY, cooldown=60)
            return "", "", f"ConnectionError: {str(e)[:100]}"

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            return "", "", f"HTTP {code}: {str(e)[:100]}"

        except ValueError as e:
            logger.warning(f"Mesh пустой ответ: {e}")
            return "", "", str(e)

        except Exception as e:
            err = str(e)
            logger.warning(f"Mesh streaming exception ({err[:80]}) → no-stream fallback")

        # ── Fallback: без streaming ────────────────────────────────────────────
        try:
            timeout = MESH_TIMEOUT_VER if role == "verifier" else MESH_TIMEOUT_GEN
            payload_ns = {**payload, "stream": False}
            resp = requests.post(
                url, json=payload_ns, headers=headers, timeout=timeout
            )
            if resp.status_code == 503:
                return "", "", "HTTP 503: Mesh (no-stream fallback)"
            resp.raise_for_status()
            data    = resp.json()
            content = _strip_think(data["choices"][0]["message"]["content"])
            model   = data.get("model", "unknown")
            logger.info(f"Mesh [{role}] no-stream fallback OK | model={model}")
            return content, model, ""
        except Exception as e:
            return "", "", str(e)[:200]

    def _call_providers(self, providers: list, prompt: str, role: str) -> Tuple[str, str]:
        """Перебирает список провайдеров по порядку (cloud fallback)."""
        error_prefix = "Generator" if role == "generator" else "Verifier"
        for p in providers:
            if _circuit_open(p.id):
                continue
            text, t_in, t_out, err = self._call(p, prompt, role)
            if text:
                tracker.record_call(p.id, tokens_in=t_in, tokens_out=t_out)
                logger.info(
                    f"LLMClient.{role} ✓ cloud {p.label} | "
                    f"in={t_in} out={t_out}"
                )
                return text, f"{p.provider}/{p.model}"
            tracker.record_call(p.id, error=True, error_msg=err)
            logger.warning(
                f"LLMClient.{role} ✗ cloud {p.label}: {err[:150]}"
            )
            if any(kw in err for kw in ("timed out", "timeout", "503",
                                         "Service Unavailable", "ConnectionError")):
                if _is_local(p.api_base):
                    _circuit_trip(p.id)
        return f"[{error_prefix} error] all providers failed", "none"

    def _call(self, p, prompt: str, role: str):
        if p.provider == "gemini":
            return self._call_gemini(p, prompt, role)
        # local_llama → streaming, остальные → нет
        use_stream = p.provider == "local_llama"
        return self._call_openai_compat(p, prompt, role, use_stream=use_stream)

    def _call_openai_compat(
        self, p, prompt: str, role: str, use_stream: bool = False
    ) -> Tuple[str, int, int, str]:
        url = f"{p.api_base}/chat/completions"
        if p.provider == "huggingface":
            url = os.getenv(
                "HF_CHAT_COMPLETIONS_URL",
                "https://router.huggingface.co/v1/chat/completions",
            )
        headers     = {"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"}
        temperature = (
            0.5 if p.provider in ("huggingface", "nvidia")
            else (0.3 if role == "verifier" else 0.7)
        )
        max_tokens = 2048 if role == "verifier" else 4096
        timeout    = getattr(p, "timeout", None) or _get_timeout(p.api_base, role)
        streaming  = use_stream or _is_local(p.api_base)

        payload = {
            "model":       p.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      streaming,
        }

        if streaming:
            try:
                full_text:     list[str] = []
                first_ok:      bool      = False
                last_chunk_at: float     = time.time()
                t0 = last_chunk_at

                with requests.post(
                    url, json=payload, headers=headers,
                    stream=True, timeout=STREAM_TOTAL_TIMEOUT,
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        now = time.time()
                        if now - last_chunk_at > STREAM_IDLE_TIMEOUT:
                            raise TimeoutError(
                                f"local idle {STREAM_IDLE_TIMEOUT}s"
                            )
                        token = _parse_sse_line(raw_line)
                        if token == '\x00':
                            break
                        if token:
                            full_text.append(token)
                            last_chunk_at = now
                            if not first_ok:
                                first_ok = True
                                logger.info(
                                    f"{p.label} первый токен за "
                                    f"{now - t0:.1f}s"
                                )

                final = _strip_think(''.join(full_text))
                if not final.strip():
                    raise ValueError("local_llama: пустой streaming-ответ")

                t_in  = _estimate_tokens(prompt)
                t_out = _estimate_tokens(final)
                return final, t_in, t_out, ""

            except Exception as e:
                logger.warning(
                    f"{p.label} streaming failed ({str(e)[:60]})"
                    " — no-stream fallback"
                )
                payload["stream"] = False

        # ── Обычный запрос ────────────────────────────────────────────────────
        try:
            payload["stream"] = False
            resp    = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data    = resp.json()
            content = _strip_think(data["choices"][0]["message"]["content"])
            usage   = data.get("usage", {})
            t_in    = usage.get("prompt_tokens",     _estimate_tokens(prompt))
            t_out   = usage.get("completion_tokens", _estimate_tokens(content or ""))
            return content, t_in, t_out, ""
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_gemini(self, p, prompt: str, role: str):
        # ── без изменений из v4.5.3 ──────────────────────────────────────────
        url = (
            f"{p.api_base}/models/{p.model}:generateContent"
            f"?key={p.api_key}"
        )
        max_output = 4096 if role == "verifier" else 1024
        timeout    = getattr(p, "timeout", None) or TIMEOUT_REMOTE_VER
        payload    = {
            "contents":         [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output},
        }
        try:
            resp  = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data  = resp.json()
            text  = _strip_think(data["candidates"][0]["content"]["parts"][0]["text"])
            usage = data.get("usageMetadata", {})
            t_in  = usage.get("promptTokenCount",     _estimate_tokens(prompt))
            t_out = usage.get("candidatesTokenCount", _estimate_tokens(text or ""))
            return text, t_in, t_out, ""
        except Exception as e:
            return "", 0, 0, str(e)[:200]
