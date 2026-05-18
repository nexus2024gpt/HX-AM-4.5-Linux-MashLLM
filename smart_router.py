# smart_router.py — HX-AM v4.5.1
"""
Smart Router — адаптивная маршрутизация LLM-запросов.

Почему НЕ используем Qwen3-0.8B для роутинга:
  - +500ms на каждый запрос только для выбора провайдера
  - Решает задачу выбора task complexity (дешёвая vs дорогая модель),
    а не task routing (кто ДОСТУПЕН прямо сейчас)
  - Для HX-AM нужен именно второй вариант

Что делает этот роутер:
  1. HealthChecker   — фоновый поток, пингует провайдеров каждые 30s
  2. ProviderMetrics — EMA-латентность, success rate, streak счётчики
  3. CircuitBreaker  — мгновенная блокировка на N секунд после фейла
  4. SmartRouter     — ранжирует провайдеров по score, возвращает
                       отсортированный список без дополнительных запросов

Формула score:
  score = success_rate² × 0.40
        + latency_score  × 0.35   (1 / (1 + norm_latency))
        + priority_score × 0.15   (1 / provider.priority)
        + freshness      × 0.10   (бонус за недавнее неиспользование)

Интеграция: заменяет tracker.get_providers_for_role() в llm_client_v_4.py
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("HXAM.router")

# ── Константы ────────────────────────────────────────────────────────────────

# Circuit Breaker
CB_COOLDOWN_TIMEOUT  = 120   # сек блокировки после таймаута (было 180, снижаем)
CB_COOLDOWN_ERROR    = 60    # сек блокировки после HTTP-ошибки
CB_TRIP_CONSECUTIVE  = 2     # сколько последовательных ошибок → trip

# Health Check
HC_INTERVAL          = 30    # сек между проверками
HC_TIMEOUT_LOCAL     = 5     # сек на пинг localhost
HC_TIMEOUT_REMOTE    = 10    # сек на пинг внешнего API

# Таймауты запросов
CALL_TIMEOUT_LOCAL_GEN   = 30   # localhost генератор
CALL_TIMEOUT_LOCAL_VER   = 45   # localhost верификатор
CALL_TIMEOUT_REMOTE_GEN  = 60   # внешний генератор
CALL_TIMEOUT_REMOTE_VER  = 90   # внешний верификатор

# EMA
EMA_ALPHA = 0.25   # скорость обновления EMA latency (0.25 = ~4 последних вызова)

# Нормализация латентности (секунды → score)
LATENCY_REF = 5.0   # базовая "хорошая" латентность; при ней score=0.5


# ── Метрики провайдера ────────────────────────────────────────────────────────

@dataclass
class ProviderMetrics:
    provider_id:      str
    latency_ema:      float = 3.0    # начальная EMA (оптимистичный старт)
    success_count:    int   = 0
    failure_count:    int   = 0
    consecutive_fail: int   = 0
    consecutive_ok:   int   = 0
    last_used:        float = 0.0
    last_success:     float = 0.0
    last_failure:     float = 0.0
    last_error_msg:   str   = ""
    is_healthy:       bool  = True   # результат последнего health check
    # Скользящее окно последних 20 вызовов (True=ok, False=fail)
    _window:          deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def total_calls(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate по скользящему окну (последние 20 вызовов)."""
        if not self._window:
            return 1.0   # новый провайдер — оптимистичный старт
        return sum(self._window) / len(self._window)

    @property
    def latency_score(self) -> float:
        """Нормализованный score латентности [0, 1]. Меньше задержка → выше score."""
        return 1.0 / (1.0 + self.latency_ema / LATENCY_REF)

    @property
    def freshness_bonus(self) -> float:
        """Небольшой бонус если провайдер давно не использовался (load balance)."""
        idle_secs = time.time() - max(self.last_used, 0.001)
        return min(0.1, idle_secs / 600.0)   # макс +0.1 за 10+ минут простоя

    def record_success(self, latency: float):
        self.success_count    += 1
        self.consecutive_ok   += 1
        self.consecutive_fail  = 0
        self.last_success      = time.time()
        self.last_used         = time.time()
        self._window.append(True)
        # EMA обновление
        self.latency_ema = EMA_ALPHA * latency + (1 - EMA_ALPHA) * self.latency_ema

    def record_failure(self, error_msg: str = ""):
        self.failure_count    += 1
        self.consecutive_fail += 1
        self.consecutive_ok    = 0
        self.last_failure      = time.time()
        self.last_used         = time.time()
        self.last_error_msg    = error_msg[:120]
        self._window.append(False)

    def compute_score(self, priority: int) -> float:
        """Итоговый score для ранжирования. Выше = лучше."""
        sr = self.success_rate
        ls = self.latency_score
        ps = 1.0 / max(priority, 1)   # priority=1 → ps=1.0, priority=10 → ps=0.1
        fb = self.freshness_bonus
        return (sr ** 2) * 0.40 + ls * 0.35 + ps * 0.15 + fb * 0.10


# ── Circuit Breaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    """Per-провайдер circuit breaker с логикой half-open."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._blocked: Dict[str, float] = {}   # provider_id → unblock_timestamp

    def is_open(self, provider_id: str) -> bool:
        """True если провайдер заблокирован."""
        with self._lock:
            until = self._blocked.get(provider_id, 0)
            if until > time.time():
                return True
            if provider_id in self._blocked:
                del self._blocked[provider_id]
                logger.info(f"[CB] {provider_id} — circuit CLOSED (cooldown expired)")
            return False

    def trip(self, provider_id: str, cooldown: int):
        with self._lock:
            self._blocked[provider_id] = time.time() + cooldown
        logger.warning(f"[CB] {provider_id} — circuit OPEN для {cooldown}s")

    def reset(self, provider_id: str):
        with self._lock:
            self._blocked.pop(provider_id, None)

    def status(self) -> Dict[str, float]:
        now = time.time()
        with self._lock:
            return {pid: round(until - now, 1) for pid, until in self._blocked.items() if until > now}


# ── Health Checker ────────────────────────────────────────────────────────────

class HealthChecker:
    """
    Фоновый поток, который периодически пингует провайдеров.
    Для локальных — GET /v1/models (быстро, без LLM-вызова).
    Для внешних  — HEAD-запрос к base URL.
    """

    def __init__(self, metrics_store: Dict[str, ProviderMetrics], cb: CircuitBreaker):
        self._metrics = metrics_store
        self._cb      = cb
        self._thread  = None
        self._stop    = threading.Event()
        self._lock    = threading.Lock()
        self._providers_snapshot: list = []

    def start(self, providers: list):
        """Запуск фонового потока. providers — список ProviderConfig."""
        with self._lock:
            self._providers_snapshot = list(providers)
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="HX-HealthChecker"
        )
        self._thread.start()
        logger.info("HealthChecker запущен (интервал %ds)", HC_INTERVAL)

    def update_providers(self, providers: list):
        with self._lock:
            self._providers_snapshot = list(providers)

    def stop(self):
        self._stop.set()

    def check_now(self, provider_id: str, api_base: str, api_key: str = "") -> bool:
        """Синхронная проверка одного провайдера. True = доступен."""
        return self._check_single(provider_id, api_base, api_key)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.wait(HC_INTERVAL):
            with self._lock:
                snapshot = list(self._providers_snapshot)
            for p in snapshot:
                pid      = p.get("id", "")
                api_base = p.get("api_base", "")
                api_key  = p.get("api_key", "")
                if not pid or not api_base:
                    continue
                healthy = self._check_single(pid, api_base, api_key)
                m = self._metrics.get(pid)
                if m:
                    m.is_healthy = healthy
                # Если стал здоровым — снимаем CB (если он не от таймаута)
                if healthy and self._cb.is_open(pid):
                    self._cb.reset(pid)
                    logger.info(f"HealthChecker: {pid} восстановлен → CB reset")

    def _check_single(self, provider_id: str, api_base: str, api_key: str) -> bool:
        is_local = "localhost" in api_base or "127.0.0.1" in api_base
        timeout  = HC_TIMEOUT_LOCAL if is_local else HC_TIMEOUT_REMOTE
        try:
            if is_local:
                # Лёгкий endpoint без LLM-вызова
                url = f"{api_base}/models"
                r   = requests.get(url, timeout=timeout,
                                   headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
                ok  = r.status_code in (200, 404)   # 404 тоже значит сервер жив
            elif "generativelanguage.googleapis.com" in api_base:
                # Gemini: просто проверяем достижимость
                r  = requests.head(api_base, timeout=timeout)
                ok = r.status_code < 500
            else:
                r  = requests.head(api_base, timeout=timeout)
                ok = r.status_code < 500

            logger.debug(f"HealthCheck {provider_id}: {'✓' if ok else '✗'} ({getattr(r,'status_code','?')})")
            return ok
        except Exception as e:
            logger.debug(f"HealthCheck {provider_id}: ✗ {str(e)[:60]}")
            return False


# ── Smart Router ──────────────────────────────────────────────────────────────

class SmartRouter:
    """
    Основной класс маршрутизации.

    Использование:
        router = SmartRouter()
        router.init(tracker)                     # один раз при старте

        providers = router.rank(role="generator")  # список ProviderConfig, лучший первый
        for p in providers:
            ...  # вызов LLM
            router.record_result(p.id, latency, success=True/False, error_msg)
    """

    _instance: Optional["SmartRouter"] = None

    def __init__(self):
        self._metrics:  Dict[str, ProviderMetrics] = {}
        self._cb        = CircuitBreaker()
        self._hc        = HealthChecker(self._metrics, self._cb)
        self._tracker   = None
        self._lock      = threading.Lock()
        self._ready     = False

    @classmethod
    def get(cls) -> "SmartRouter":
        """Singleton доступ."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(self, tracker_instance):
        """
        Инициализация из APIUsageTracker.
        Вызывать один раз при старте сервера.
        """
        self._tracker = tracker_instance
        self._sync_from_tracker()
        providers_raw = tracker_instance.get_providers()
        self._hc.start(providers_raw)
        self._ready = True

        logger.info(
            f"SmartRouter инициализирован: "
            f"{len(self._metrics)} провайдеров, "
            f"HealthChecker активен"
        )

    def rank(self, role: str = "generator") -> list:
        """
        Возвращает провайдеров для роли, отсортированных по score.
        Заблокированные CB провайдеры идут в конец (не исключаются —
        если ВСЕ заблокированы, даём шанс попробовать хотя бы один).
        """
        if not self._ready or not self._tracker:
            # Фолбэк: старая логика tracker
            return self._tracker.get_providers_for_role(role) if self._tracker else []

        self._sync_from_tracker()
        candidates = self._tracker.get_providers_for_role(role)
        if not candidates:
            return []

        scored: List[Tuple[float, bool, object]] = []
        for p in candidates:
            m  = self._get_or_create_metrics(p.id)
            cb = self._cb.is_open(p.id)
            score = m.compute_score(p.priority)
            if cb:
                score -= 1000   # штраф, но не исключение
            scored.append((score, p))

        scored.sort(key=lambda x: -x[0])
        ranked = [p for _, p in scored]

        if logger.isEnabledFor(logging.DEBUG):
            for s, p in scored[:3]:
                m = self._metrics.get(p.id)
                logger.debug(
                    f"  {p.id}: score={s:.3f} "
                    f"sr={m.success_rate:.2f} lat={m.latency_ema:.1f}s"
                    if m else f"  {p.id}: score={s:.3f}"
                )

        return ranked

    def record_result(
        self,
        provider_id: str,
        latency: float,
        success: bool,
        error_msg: str = "",
        role: str = "",
    ):
        """Записывает результат вызова и обновляет метрики + CB."""
        m = self._get_or_create_metrics(provider_id)

        if success:
            m.record_success(latency)
            self._cb.reset(provider_id)
            logger.debug(f"[Router] {provider_id} ✓ {latency:.2f}s | "
                         f"sr={m.success_rate:.2f} ema={m.latency_ema:.2f}s")
        else:
            m.record_failure(error_msg)
            # Определяем тип фейла
            is_timeout = any(kw in error_msg for kw in
                             ("timed out", "timeout", "Timeout"))
            is_503     = "503" in error_msg or "Service Unavailable" in error_msg
            is_conn    = "ConnectionError" in error_msg or "Connection refused" in error_msg

            p_config = self._get_provider_config(provider_id)
            is_local  = p_config and ("localhost" in p_config.get("api_base", "")
                                      or "127.0.0.1" in p_config.get("api_base", ""))

            if m.consecutive_fail >= CB_TRIP_CONSECUTIVE:
                cooldown = CB_COOLDOWN_TIMEOUT if (is_timeout or is_local) else CB_COOLDOWN_ERROR
                self._cb.trip(provider_id, cooldown)

            logger.warning(
                f"[Router] {provider_id} ✗ "
                f"(fail #{m.consecutive_fail}) {error_msg[:80]}"
            )

    def get_call_timeout(self, provider_id: str, role: str) -> int:
        """Возвращает таймаут для конкретного вызова."""
        p = self._get_provider_config(provider_id)
        api_base = (p or {}).get("api_base", "")
        is_local = "localhost" in api_base or "127.0.0.1" in api_base
        if is_local:
            return CALL_TIMEOUT_LOCAL_VER if role == "verifier" else CALL_TIMEOUT_LOCAL_GEN
        return CALL_TIMEOUT_REMOTE_VER if role == "verifier" else CALL_TIMEOUT_REMOTE_GEN

    def status(self) -> dict:
        """Диагностическая сводка для /router/status endpoint."""
        self._sync_from_tracker()
        providers_info = []
        for pid, m in sorted(self._metrics.items(), key=lambda x: -x[1].success_rate):
            p = self._get_provider_config(pid)
            priority = (p or {}).get("priority", 99)
            score    = m.compute_score(priority)
            cb_open  = self._cb.is_open(pid)
            providers_info.append({
                "id":               pid,
                "label":            (p or {}).get("label", pid),
                "score":            round(score, 3),
                "success_rate":     round(m.success_rate, 3),
                "latency_ema_s":    round(m.latency_ema, 2),
                "consecutive_fail": m.consecutive_fail,
                "total_calls":      m.total_calls,
                "is_healthy":       m.is_healthy,
                "circuit_open":     cb_open,
                "circuit_blocked_s": round(max(0, self._cb.status().get(pid, 0)), 0),
                "last_error":       m.last_error_msg,
            })
        return {
            "providers":    providers_info,
            "circuit_state": self._cb.status(),
            "hc_running":   self._hc._thread.is_alive() if self._hc._thread else False,
        }

    def force_health_check(self, provider_id: str) -> bool:
        """Немедленная проверка провайдера (для /router/check/{id})."""
        p = self._get_provider_config(provider_id)
        if not p:
            return False
        ok = self._hc.check_now(provider_id, p.get("api_base", ""), p.get("api_key", ""))
        m  = self._get_or_create_metrics(provider_id)
        m.is_healthy = ok
        return ok

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sync_from_tracker(self):
        """Синхронизирует метрики с актуальным списком провайдеров из tracker."""
        if not self._tracker:
            return
        providers = self._tracker.get_providers()
        with self._lock:
            for p in providers:
                pid = p.get("id", "")
                if pid and pid not in self._metrics:
                    self._metrics[pid] = ProviderMetrics(provider_id=pid)
        # Обновляем snapshot для HealthChecker
        self._hc.update_providers(providers)

    def _get_or_create_metrics(self, provider_id: str) -> ProviderMetrics:
        with self._lock:
            if provider_id not in self._metrics:
                self._metrics[provider_id] = ProviderMetrics(provider_id=provider_id)
            return self._metrics[provider_id]

    def _get_provider_config(self, provider_id: str) -> Optional[dict]:
        if not self._tracker:
            return None
        for p in self._tracker.get_providers():
            if p.get("id") == provider_id:
                return p
        return None


# ── Singleton инициализация ───────────────────────────────────────────────────

smart_router = SmartRouter.get()
