# llm_client_v_4.py — HX-AM v4.5.3 [Mesh auto-routing only]

import logging
import os
import time
from threading import Lock
from typing import Dict, Tuple

import requests
from api_usage_tracker import tracker

logger = logging.getLogger("HXAM.llm")

TIMEOUT_LOCAL      = 240
TIMEOUT_REMOTE_GEN = 60
TIMEOUT_REMOTE_VER = 90
MESH_TIMEOUT_GEN   = int(os.getenv("MASH_CALL_TIMEOUT_GEN", "300"))
MESH_TIMEOUT_VER   = int(os.getenv("MASH_CALL_TIMEOUT_VER", "300"))

CIRCUIT_COOLDOWN    = 180
_circuit_lock       = Lock()
_circuit_state: Dict[str, float] = {}

_MESH_BASE = os.getenv("MASH_BASE_URL", "http://localhost:9337")

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

def _circuit_trip(provider_id: str):
    with _circuit_lock:
        _circuit_state[provider_id] = time.time() + CIRCUIT_COOLDOWN
    logger.warning(f"[CB] {provider_id} отключён на {CIRCUIT_COOLDOWN}s")

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:
    def generate(self, prompt: str) -> Tuple[str, str]:
        # 1. Локальный провайдер (local_llama)
        local_providers = [p for p in tracker.get_providers_for_role("generator") if p.provider == "local_llama"]
        if local_providers:
            p = local_providers[0]
            text, t_in, t_out, err = self._call(p, prompt, role="generator")
            if text:
                tracker.record_call(p.id, tokens_in=t_in, tokens_out=t_out)
                logger.info(f"LLMClient.generator ✓ {p.label} | in={t_in} out={t_out}")
                return text, f"local/{p.model}"
            logger.warning(f"Local LLM failed: {err}")

        # 2. Mesh auto (без указания модели)
        text, _, err = self._call_mesh_auto(prompt, role="generator")
        if text:
            logger.info(f"LLMClient.generator ✓ Mesh[auto] | in={_estimate_tokens(prompt)} out={_estimate_tokens(text)}")
            return text, "mesh/auto"
        logger.warning(f"LLMClient.generator: Mesh auto failed: {err}")

        # 3. Остальные облачные провайдеры (исключая local_llama и mesh)
        other_providers = [p for p in tracker.get_providers_for_role("generator") if p.provider not in ("local_llama", "mesh")]
        if not other_providers:
            return "[Generator error] no providers configured", "none"
        return self._call_providers(other_providers, prompt, role="generator")

    def verify(self, statement: str, context: str = "") -> Tuple[str, str]:
        full_prompt = f"Context: {context}\n\n{statement}" if context else statement

        # 1. Локальный провайдер
        local_providers = [p for p in tracker.get_providers_for_role("verifier") if p.provider == "local_llama"]
        if local_providers:
            p = local_providers[0]
            text, t_in, t_out, err = self._call(p, full_prompt, role="verifier")
            if text:
                tracker.record_call(p.id, tokens_in=t_in, tokens_out=t_out)
                logger.info(f"LLMClient.verifier ✓ {p.label} | in={t_in} out={t_out}")
                return text, f"local/{p.model}"
            logger.warning(f"Local LLM failed: {err}")

        # 2. Mesh auto
        text, _, err = self._call_mesh_auto(full_prompt, role="verifier")
        if text:
            logger.info(f"LLMClient.verifier ✓ Mesh[auto] | in={_estimate_tokens(full_prompt)} out={_estimate_tokens(text)}")
            return text, "mesh/auto"
        logger.warning(f"LLMClient.verifier: Mesh auto failed: {err}")

        # 3. Остальные облачные провайдеры
        other_providers = [p for p in tracker.get_providers_for_role("verifier") if p.provider not in ("local_llama", "mesh")]
        if not other_providers:
            return "[Verifier error] no providers configured", "none"
        return self._call_providers(other_providers, full_prompt, role="verifier")

    def _call_mesh_auto(self, prompt: str, role: str) -> Tuple[str, str, str]:
        url = f"{_MESH_BASE}/v1/chat/completions"
        timeout = MESH_TIMEOUT_VER if role == "verifier" else MESH_TIMEOUT_GEN
        temperature = 0.3 if role == "verifier" else 0.7
        # Для генератора нужно больше места для JSON + возможной преамбулы
        if role == "generator":
            max_tokens = 4096   # достаточно для полной гипотезы
        else:
            max_tokens = 2048   # верификатору обычно хватает
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        logger.debug(f"Calling MeshLLM auto, role={role}, prompt_len={len(prompt)}")
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 503:
                return "", "", "HTTP 503: Mesh недоступен"
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            used_model = data.get("model", "unknown")
            return content, used_model, ""
        except Exception as e:
            return "", "", str(e)[:200]

    def _call_providers(self, providers: list, prompt: str, role: str) -> Tuple[str, str]:
        for p in providers:
            if _circuit_open(p.id):
                continue
            text, t_in, t_out, err = self._call(p, prompt, role)
            if text:
                tracker.record_call(p.id, tokens_in=t_in, tokens_out=t_out)
                logger.info(f"LLMClient.{role} ✓ {p.label} | in={t_in} out={t_out}")
                return text, f"{p.provider}/{p.model}"
            tracker.record_call(p.id, error=True, error_msg=err)
            logger.warning(f"LLMClient.{role} ✗ {p.label}: {err[:150]}")
            if any(kw in err for kw in ("timed out", "timeout", "503", "Service Unavailable", "ConnectionError")):
                if _is_local(p.api_base):
                    _circuit_trip(p.id)
        return (f"[{role.capitalize()} error] all providers failed", "none")

    def _call(self, p, prompt: str, role: str):
        if p.provider == "gemini":
            return self._call_gemini(p, prompt, role)
        else:
            return self._call_openai_compat(p, prompt, role)

    def _call_openai_compat(self, p, prompt: str, role: str):
        url = f"{p.api_base}/chat/completions"
        if p.provider == "huggingface":
            url = os.getenv("HF_CHAT_COMPLETIONS_URL", "https://router.huggingface.co/v1/chat/completions")
        headers = {"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"}
        temperature = 0.5 if p.provider in ("huggingface", "nvidia") else (0.3 if role == "verifier" else 0.7)
        max_tokens = 2048 if role == "verifier" else 1024
        timeout = getattr(p, "timeout", None) or _get_timeout(p.api_base, role)
        payload = {"model": p.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            t_in = usage.get("prompt_tokens", _estimate_tokens(prompt))
            t_out = usage.get("completion_tokens", _estimate_tokens(content or ""))
            return content, t_in, t_out, ""
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_gemini(self, p, prompt: str, role: str):
        url = f"{p.api_base}/models/{p.model}:generateContent?key={p.api_key}"
        max_output = 4096 if role == "verifier" else 1024
        timeout = getattr(p, "timeout", None) or TIMEOUT_REMOTE_VER
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_output}}
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            t_in = usage.get("promptTokenCount", _estimate_tokens(prompt))
            t_out = usage.get("candidatesTokenCount", _estimate_tokens(text or ""))
            return text, t_in, t_out, ""
        except Exception as e:
            return "", 0, 0, str(e)[:200]
