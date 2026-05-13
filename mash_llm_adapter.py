# mash_llm_adapter.py — HX-AM v4.5.3
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("HXAM.mash_adapter")

# ── Конфиг из .env ────────────────────────────────────────────────────────────
MASH_BASE_URL      = os.getenv("MASH_BASE_URL",      "http://localhost:9337")
MASH_MGMT_URL      = os.getenv("MASH_MGMT_URL",      "http://localhost:3131")
MASH_POLL_INTERVAL = float(os.getenv("MASH_POLL_INTERVAL", "10"))
MASH_USE_SSE       = os.getenv("MASH_USE_SSE", "true").lower() == "true"
MASH_TIMEOUT       = float(os.getenv("MASH_TIMEOUT", "5"))
MASH_SSE_TIMEOUT   = float(os.getenv("MASH_SSE_TIMEOUT", "30"))

# Минимальный RAM для роли (GB). Модели ниже порога идут в конец очереди.
MASH_MIN_RAM_GENERATOR = float(os.getenv("MASH_MIN_RAM_GENERATOR", "8"))
MASH_MIN_RAM_VERIFIER  = float(os.getenv("MASH_MIN_RAM_VERIFIER",  "6"))

_DEFAULT_GEN_HINTS = os.getenv(
    "MASH_GEN_HINTS", "qwen,llama,mistral,phi,gemma,glm"
).lower().split(",")

_DEFAULT_VER_HINTS = os.getenv(
    "MASH_VER_HINTS", "qwen,llama,mistral,phi,gemma,glm"
).lower().split(",")

# Regex для очистки GGUF-суффиксов из имени модели
_GGUF_SUFFIX_RE = re.compile(
    r"[-_](Q[0-9]+_K_[MS]|IQ[0-9]+_[A-Z]+|f16|bf16|fp16|"
    r"int8|int4|GPTQ|AWQ|GGUF|gguf)$",
    re.IGNORECASE,
)


def _strip_gguf_suffix(name: str) -> str:
    """
    'Qwen3-4B-Q4_K_M'        → 'Qwen3-4B'
    'gemma-4-E4B-it-Q4_K_M'  → 'gemma-4-E4B-it'
    'GLM-4.7-Flash'           → 'GLM-4.7-Flash'  (без изменений)
    """
    result = _GGUF_SUFFIX_RE.sub("", name).rstrip("-_")
    return result if result else name


def _parse_ram_gb(raw: Any) -> float:
    """
    Парсит объём памяти из разных форматов:
    '38.7 GB' | '38700 MB' | '38700000000' | 38.7
    """
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        v = float(raw)
        # Если значение > 1000 — скорее всего байты или MB
        if v > 1_000_000:
            return round(v / 1_073_741_824, 1)   # bytes → GB
        if v > 1_000:
            return round(v / 1024, 1)              # MB → GB
        return round(v, 1)                         # уже GB
    s = str(raw).strip().lower()
    match = re.search(r"([\d.]+)\s*(gb|mb|kb|b)?", s)
    if not match:
        return 0.0
    val  = float(match.group(1))
    unit = match.group(2) or "gb"
    return {
        "gb": val,
        "mb": round(val / 1024, 1),
        "kb": round(val / 1_048_576, 1),
        "b":  round(val / 1_073_741_824, 1),
    }.get(unit, val)


def _parse_latency_ms(raw: Any) -> float:
    """'290 ms' | 290 | '290ms' | None → float ms (0 если нет данных)"""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower().replace("ms", "").replace("—", "0")
    try:
        return float(s.strip())
    except ValueError:
        return 0.0


# ── Модель состояния ──────────────────────────────────────────────────────────

@dataclass
class MashModel:
    """Состояние одной модели/ноды в Mesh."""
    id:              str             # полное имя из API (может содержать GGUF-суффикс)
    name:            str             # display name
    ready:           bool   = False
    queue_length:    int    = 0
    active_requests: int    = 0
    load:            float  = 0.0   # 0.0–1.0
    ram_gb:          float  = 0.0   # ОЗУ ноды (GB) — ключевой критерий силы модели
    latency_ms:      float  = 0.0   # задержка ответа (мс), 0 = нет данных
    node_id:         str    = ""
    node_role:       str    = ""    # Host / Worker
    raw:             Dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_name(self) -> str:
        """Имя без GGUF-суффикса: 'Qwen3-4B-Q4_K_M' → 'Qwen3-4B'."""
        return _strip_gguf_suffix(self.name)

    @property
    def canonical_id(self) -> str:
        """ID без GGUF-суффикса."""
        return _strip_gguf_suffix(self.id)

    def all_name_variants(self) -> List[str]:
        """
        Все варианты имени для перебора при HTTP 400.
        Порядок: короткое → полное → node_id.
        """
        variants: List[str] = []
        canonical = self.canonical_name
        if canonical:
            variants.append(canonical)
        if self.id and self.id != canonical:
            variants.append(self.id)
        if self.name and self.name not in variants:
            variants.append(self.name)
        if self.node_id and self.node_id not in variants:
            variants.append(self.node_id)
        # Убираем дубли, сохраняя порядок
        seen = set()
        result = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                result.append(v)
        return result

    def score(self) -> float:
        """
        Рейтинг ноды — чем МЕНЬШЕ, тем ЛУЧШЕ (для min-sort).

        Формула:
          base   = queue*2 + active + load*3
          ram    = -ram_gb * 0.15      (бонус за RAM: +38GB → -5.7 очков)
          lat    = latency_ms * 0.005  (штраф за задержку: 300ms → +1.5)
          result = base + ram_bonus + lat_penalty
        """
        base    = self.queue_length * 2 + self.active_requests + self.load * 3
        ram_bonus  = -(self.ram_gb * 0.15)         # больше RAM → ниже score
        lat_penalty = self.latency_ms * 0.005       # меньше latency → ниже score
        return round(base + ram_bonus + lat_penalty, 3)

    def matches_role_hint(self, role: str) -> bool:
        """Мягкий hint-фильтр. Не исключает — только для приоритизации."""
        hints = _DEFAULT_GEN_HINTS if role == "generator" else _DEFAULT_VER_HINTS
        name_lower = self.canonical_name.lower()
        return any(h in name_lower for h in hints) if hints else True

    def matches_role(self, role: str) -> bool:
        """Алиас для matches_role_hint (используется в трекере)."""
        return self.matches_role_hint(role)

    @classmethod
    def from_status_dict(
        cls, data: Dict[str, Any], node_id: str = "", node_role: str = ""
    ) -> "MashModel":
        model_id = (
            data.get("id") or data.get("model_id")
            or data.get("name") or "unknown"
        )
        name = data.get("name") or data.get("model") or model_id

        # Статус готовности
        ready_raw = data.get("ready", data.get("status", ""))
        if isinstance(ready_raw, bool):
            ready = ready_raw
        else:
            ready = str(ready_raw).lower() in (
                "true", "ready", "1", "loaded", "running", "serving"
            )

        queue  = int(data.get("queue_length", data.get("queue", 0)) or 0)
        active = int(data.get("active_requests", data.get("active", 0)) or 0)
        load_v = float(data.get("load", 0.0) or 0.0)
        if load_v > 1.0:
            load_v = load_v / 100.0  # % → доля

        # RAM — пробуем несколько ключей
        ram_raw = (
            data.get("ram_gb") or data.get("memory_gb")
            or data.get("ram") or data.get("memory")
            or data.get("total_memory") or data.get("vram_gb")
            or 0
        )
        ram_gb = _parse_ram_gb(ram_raw)

        # Latency
        lat_raw = (
            data.get("latency_ms") or data.get("latency")
            or data.get("response_time") or data.get("avg_latency")
            or 0
        )
        latency_ms = _parse_latency_ms(lat_raw)

        return cls(
            id=model_id, name=name, ready=ready,
            queue_length=queue, active_requests=active, load=load_v,
            ram_gb=ram_gb, latency_ms=latency_ms,
            node_id=node_id or str(data.get("node_id", "")),
            node_role=node_role or str(data.get("role", "")),
            raw=data,
        )

    @classmethod
    def from_openai_dict(cls, data: Dict[str, Any]) -> "MashModel":
        """
        Парсит запись из /v1/models.
        Пытается извлечь имя ноды из поля 'owned_by' или 'metadata'.
        """
        model_id = data.get("id", "unknown")

        # Некоторые версии MashLLM кладут доп. поля в metadata/details
        meta = data.get("metadata") or data.get("details") or {}
        ram_gb     = _parse_ram_gb(meta.get("ram_gb") or meta.get("memory"))
        latency_ms = _parse_latency_ms(meta.get("latency_ms") or meta.get("latency"))
        node_id    = str(meta.get("node_id", ""))
        node_role  = str(meta.get("role", ""))

        return cls(
            id=model_id, name=model_id,
            ready=True,
            ram_gb=ram_gb, latency_ms=latency_ms,
            node_id=node_id, node_role=node_role,
            raw=data,
        )


# ── Адаптер ───────────────────────────────────────────────────────────────────

class MashLLMAdapter:
    """
    Singleton-адаптер. Поддерживает актуальный кэш healthy-моделей MashLLM.
    Сортирует модели по RAM + latency + load.
    """

    _instance: Optional["MashLLMAdapter"] = None
    _lock: Lock = Lock()

    def __init__(self):
        self._models:       Dict[str, MashModel] = {}
        self._cache_lock:   Lock  = Lock()
        self._last_update:  float = 0.0
        self._healthy:      bool  = False
        self._stop_event:   Event = Event()
        self._poll_thread:  Optional[Thread] = None
        self._sse_thread:   Optional[Thread] = None
        self._last_top_str: Optional[str] = None

    @classmethod
    def get(cls) -> "MashLLMAdapter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ── Публичный API ─────────────────────────────────────────────

    def start(self):
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._stop_event.clear()

        try:
            self._refresh_from_status()
        except Exception as e:
            logger.warning(f"MashAdapter: начальное обновление не удалось — {e}")

        self._poll_thread = Thread(
            target=self._poll_loop, daemon=True, name="MashAdapter-Poll"
        )
        self._poll_thread.start()

        if MASH_USE_SSE:
            self._sse_thread = Thread(
                target=self._sse_loop, daemon=True, name="MashAdapter-SSE"
            )
            self._sse_thread.start()

        logger.info(
            f"MashAdapter: запущен "
            f"(poll={MASH_POLL_INTERVAL}s sse={'вкл' if MASH_USE_SSE else 'выкл'})"
        )

    def stop(self):
        self._stop_event.set()

    def is_healthy(self) -> bool:
        return self._healthy and bool(self._get_ready_models())

    def get_best_model(self, role: str = "generator") -> Optional[MashModel]:
        """
        Возвращает лучшую модель для роли.
        Приоритет: RAM ↓ → latency ↓ → queue ↓ → load ↓

        Алгоритм:
          1. Все ready-модели
          2. Сортировка по score() (учитывает RAM, latency, load)
          3. Логируем топ-3 для прозрачности выбора
        """
        ready = self._get_ready_models()
        if not ready:
            return None

        # Сортируем по score (меньше = лучше)
        sorted_models = sorted(ready, key=lambda m: m.score())

        if logger.isEnabledFor(logging.DEBUG):
            for i, m in enumerate(sorted_models[:3]):
                logger.debug(
                    f"MashAdapter top-{i+1}: '{m.canonical_name}' "
                    f"ram={m.ram_gb}GB lat={m.latency_ms}ms "
                    f"queue={m.queue_length} load={m.load:.0%} "
                    f"score={m.score():.2f}"
                )

        best = sorted_models[0]
        logger.info(
            f"MashAdapter.get_best_model(role={role}): "
            f"'{best.canonical_name}' выбрана "
            f"[ram={best.ram_gb}GB lat={best.latency_ms}ms "
            f"queue={best.queue_length} score={best.score():.2f}]"
        )
        return best

    def get_all_ready(self) -> List[MashModel]:
        """Возвращает все ready-модели (отсортированные по score)."""
        return self._get_ready_models()

    def get_all_ready_sorted(self) -> List[MashModel]:
        """Все ready-модели, отсортированные по убыванию силы."""
        return sorted(self._get_ready_models(), key=lambda m: m.score())

    def model_count(self) -> int:
        with self._cache_lock:
            return len(self._models)

    def status_summary(self) -> Dict[str, Any]:
        ready = self._get_ready_models()
        with self._cache_lock:
            all_models = list(self._models.values())
        sorted_all = sorted(all_models, key=lambda m: m.score())
        return {
            "healthy":      self._healthy,
            "last_update":  self._last_update,
            "total_models": len(all_models),
            "ready_models": len(ready),
            "models": [
                {
                    "id":            m.id,
                    "canonical":     m.canonical_name,
                    "ready":         m.ready,
                    "ram_gb":        m.ram_gb,
                    "latency_ms":    m.latency_ms,
                    "queue":         m.queue_length,
                    "load":          f"{m.load:.0%}",
                    "node_id":       m.node_id,
                    "node_role":     m.node_role,
                    "score":         round(m.score(), 3),
                    "name_variants": m.all_name_variants(),
                }
                for m in sorted_all
            ],
        }

    # ── Кэш ──────────────────────────────────────────────────────

    def _get_ready_models(self) -> List[MashModel]:
        with self._cache_lock:
            return [m for m in self._models.values() if m.ready]

    def _update_model(self, model: MashModel):
        with self._cache_lock:
            self._models[model.id] = model
        self._last_update = time.time()

    def _set_models(self, models: List[MashModel]):
        with self._cache_lock:
            self._models = {m.id: m for m in models}
        self._last_update = time.time()

    # ── REST poll /api/status ─────────────────────────────────────

    def _refresh_from_status(self):
        url = f"{MASH_MGMT_URL}/api/status"
        try:
            resp = requests.get(url, timeout=MASH_TIMEOUT)
            resp.raise_for_status()

            # MashLLM иногда возвращает text/plain или двойной JSON
            raw = resp.text.strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"MashAdapter: /api/status вернул не-JSON: {raw[:80]}")
                self._refresh_from_openai()
                return

            # Если после json.loads всё ещё строка — двойная сериализация
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    logger.warning(
                        f"MashAdapter: /api/status двойная строка: {data[:80]}"
                    )
                    self._refresh_from_openai()
                    return

            models = self._parse_status_response(data)
            if models:
                self._set_models(models)
                self._healthy = True
                ready_cnt = sum(1 for m in models if m.ready)
                top = sorted(
                    [m for m in models if m.ready],
                    key=lambda m: m.score()
                )[:3]
                top_str = " | ".join(
                    f"{m.canonical_name}({m.ram_gb}GB,{m.latency_ms:.0f}ms)"
                    for m in top
                )
                # Выводим топ только при изменении
                current_top_str = f"{ready_cnt}/{len(models)} serving-моделей. Топ: [{top_str}]"
                if self._last_top_str != current_top_str:
                    logger.info(current_top_str)
                    self._last_top_str = current_top_str
                else:
                    logger.debug(current_top_str)
                return

            # models == None или пустой список
            logger.warning(
                f"MashAdapter: /api/status не распознан: "
                f"type={type(data).__name__} keys={list(data.keys()) if isinstance(data, dict) else '—'}"
            )

        except requests.exceptions.ConnectionError:
            logger.debug("MashAdapter: /api/status недоступен")
        except requests.exceptions.Timeout:
            logger.warning("MashAdapter: /api/status timeout")
        except Exception as e:
            logger.warning(f"MashAdapter: /api/status ошибка — {e}")

        self._refresh_from_openai()

    def _refresh_from_openai(self):
        """
        Fallback: получает список моделей из OpenAI /v1/models.
        RAM и latency недоступны — score будет 0.0, приоритет по порядку списка.
        """
        url = f"{MASH_BASE_URL}/v1/models"
        try:
            resp = requests.get(url, timeout=MASH_TIMEOUT)
            resp.raise_for_status()

            raw = resp.text.strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"MashAdapter: /v1/models не-JSON: {raw[:80]}")
                self._healthy = False
                return

            raw_list = data.get("data", []) if isinstance(data, dict) else data
            if not isinstance(raw_list, list):
                logger.warning(
                    f"MashAdapter: /v1/models неожиданная структура: "
                    f"{type(raw_list).__name__}"
                )
                self._healthy = False
                return

            models = [
                MashModel.from_openai_dict(m)
                for m in raw_list
                if isinstance(m, dict)
            ]
            if models:
                self._set_models(models)
                self._healthy = True
                names = [m.canonical_name for m in models]
                logger.info(
                    f"MashAdapter: /v1/models — {len(models)} моделей "
                    f"(assumed ready): {names}"
                )
                return

        except requests.exceptions.ConnectionError:
            logger.debug("MashAdapter: /v1/models недоступен")
        except Exception as e:
            logger.warning(f"MashAdapter: /v1/models ошибка — {e}")

        self._healthy = False

    def _parse_status_response(self, data: Any) -> Optional[List[MashModel]]:
        """
        Разбирает ответ /api/status MashLLM v0.56–v0.65.
        Поддерживает форматы: 'models', 'nodes', 'serving_models'+'peers'.
        """
        # Защита: если строка — повторный парсинг
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return None

        if not isinstance(data, dict):
            logger.debug(f"_parse_status_response: ожидали dict, получили {type(data).__name__}")
            return None

        models = []

        # ----- 1. Локальные модели (serving_models) -----
        local_serving = data.get("serving_models")
        if isinstance(local_serving, list):
            local_ram = _parse_ram_gb(data.get("my_vram_gb") or data.get("vram_gb") or 0)
            local_node_id = data.get("node_id", "")
            local_role = data.get("node_status", "")
            local_lat = _parse_latency_ms(data.get("rtt_ms") or 0)
            for m_name in local_serving:
                if isinstance(m_name, str):
                    models.append(MashModel(
                        id=m_name, name=m_name, ready=True,
                        ram_gb=local_ram, latency_ms=local_lat,
                        node_id=local_node_id, node_role=local_role,
                    ))

        # ----- 2. Модели из peers -----
        peers = data.get("peers")
        if isinstance(peers, list):
            for peer in peers:
                if not isinstance(peer, dict):
                    continue
                if peer.get("state") != "serving":
                    continue
                peer_serving = peer.get("serving_models")
                if not isinstance(peer_serving, list):
                    continue
                peer_ram = _parse_ram_gb(peer.get("vram_gb") or peer.get("ram_gb") or 0)
                peer_lat = _parse_latency_ms(peer.get("rtt_ms") or 0)
                peer_id = peer.get("id", "")
                peer_role = peer.get("role", "")
                for m_name in peer_serving:
                    if isinstance(m_name, str):
                        models.append(MashModel(
                            id=m_name, name=m_name, ready=True,
                            ram_gb=peer_ram, latency_ms=peer_lat,
                            node_id=peer_id, node_role=peer_role,
                        ))

        if models:
            logger.debug(f"MashAdapter: получено {len(models)} serving-моделей из сети")
            return models

        # ----- 3. Fallback: старые форматы (models / nodes) -----
        raw_models = data.get("models")
        if isinstance(raw_models, list) and raw_models:
            result = []
            for m in raw_models:
                if isinstance(m, dict):
                    result.append(MashModel.from_status_dict(m))
            if result:
                return result

        raw_nodes = data.get("nodes")
        if isinstance(raw_nodes, list) and raw_nodes:
            node_models = []
            for node in raw_nodes:
                if not isinstance(node, dict):
                    continue
                node_id   = str(node.get("id", node.get("node_id", "")))
                node_role = str(node.get("role", node.get("type", "")))
                node_ram  = _parse_ram_gb(node.get("ram_gb") or node.get("memory_gb") or 0)
                node_lat  = _parse_latency_ms(node.get("latency_ms") or node.get("latency") or 0)
                for m_raw in node.get("models", []):
                    if not isinstance(m_raw, dict):
                        continue
                    m = MashModel.from_status_dict(m_raw, node_id=node_id, node_role=node_role)
                    if m.ram_gb == 0.0 and node_ram > 0:
                        m.ram_gb = node_ram
                    if m.latency_ms == 0.0 and node_lat > 0:
                        m.latency_ms = node_lat
                    node_models.append(m)
            if node_models:
                return node_models

        # ----- 4. Сам ответ — одна модель -----
        if "id" in data or "name" in data:
            return [MashModel.from_status_dict(data)]

        return None

    # ── SSE /api/events ───────────────────────────────────────────

    def _sse_loop(self):
        url = f"{MASH_MGMT_URL}/api/events"
        backoff = 1.0

        while not self._stop_event.is_set():
            try:
                logger.info(f"MashAdapter SSE: подключение к {url}")
                with requests.get(
                    url, stream=True,
                    timeout=(MASH_TIMEOUT, MASH_SSE_TIMEOUT),
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    resp.raise_for_status()
                    backoff = 1.0
                    logger.info("MashAdapter SSE: подключён")
                    for line in resp.iter_lines(decode_unicode=True):
                        if self._stop_event.is_set():
                            break
                        if line:
                            self._handle_sse_line(line)
            except requests.exceptions.ConnectionError:
                logger.debug(f"MashAdapter SSE: нет соединения, retry через {backoff:.0f}s")
            except requests.exceptions.Timeout:
                logger.debug("MashAdapter SSE: timeout")
            except Exception as e:
                logger.warning(f"MashAdapter SSE: ошибка — {e}")

            if not self._stop_event.wait(backoff):
                backoff = min(backoff * 2, 60.0)

    def _handle_sse_line(self, line: str):
        if not line.startswith("data:"):
            return
        payload_str = line[5:].strip()
        if not payload_str:
            return
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            return

        event_type = payload.get("type") or payload.get("event") or ""
        data       = payload.get("data") or payload

        if event_type in ("model_ready", "model_loaded", "model_status", "status"):
            if isinstance(data, dict):
                model = MashModel.from_status_dict(data)
                self._update_model(model)
                logger.debug(
                    f"MashAdapter SSE: '{model.canonical_name}' "
                    f"ready={model.ready} ram={model.ram_gb}GB"
                )

        elif event_type in ("model_unloaded", "model_removed"):
            if isinstance(data, dict):
                mid = data.get("id") or data.get("name")
                if mid:
                    with self._cache_lock:
                        if mid in self._models:
                            self._models[mid].ready = False
                    logger.info(f"MashAdapter SSE: '{mid}' → not-ready")

        elif event_type in ("full_update", "status_update"):
            models = self._parse_status_response(
                data if isinstance(data, dict) else payload
            )
            if models:
                self._set_models(models)
                self._healthy = True
                logger.info(
                    f"MashAdapter SSE: полный апдейт — {len(models)} моделей"
                )

    # ── Poll loop ─────────────────────────────────────────────────

    def _poll_loop(self):
        while not self._stop_event.wait(MASH_POLL_INTERVAL):
            try:
                self._refresh_from_status()
            except Exception as e:
                logger.error(f"MashAdapter poll: ошибка — {e}")


# ── Синглтон ──────────────────────────────────────────────────────────────────
mash_adapter = MashLLMAdapter.get()