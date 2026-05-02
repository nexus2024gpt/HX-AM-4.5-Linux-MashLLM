# mgap_lib/config/settings.py — MGAP Library v1.0
"""
Настройки MGAP Library через переменные окружения (.env).

Переменные:
  MGAP_DB_URL           — строка подключения к БД (default: sqlite:///mgap.db)
  MGAP_REGISTRY_PATH    — путь к mgap_registry.json (default: mgap_registry.json)
  MGAP_ARTIFACTS_DIR    — папка артефактов HX-AM (default: artifacts)
  MGAP_LOG_LEVEL        — уровень логирования (default: INFO)
  MGAP_USE_LLM          — улучшать blind_spot через LLM (default: true)
  MGAP_DEFAULT_TOP_K    — количество лучших матчей (default: 3)
  MGAP_MIN_RESONANCE    — минимальный резонанс для batch (default: 0.3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MGAPSettings:
    # База данных
    db_url:           str  = "sqlite:///mgap.db"

    # Пути
    registry_path:    Path = Path("mgap_registry.json")
    artifacts_dir:    Path = Path("artifacts")
    results_dir:      Path = Path("mgap_results")
    domain_map_path:  Path = Path(__file__).parent.parent / "data" / "domain_map.json"
    unesco_tax_path:  Path = Path(__file__).parent.parent / "data" / "unesco_taxonomy.json"

    # Поведение
    use_llm:          bool  = True
    default_top_k:    int   = 3
    min_resonance:    float = 0.3
    math_type_only:   bool  = True   # только совпадение math_type при поиске
    gap_mode:         str   = "max"  # max | mean | rms

    # Логирование
    log_level:        str  = "INFO"

    @classmethod
    def from_env(cls) -> "MGAPSettings":
        """Читает настройки из переменных окружения."""
        s = cls()
        if v := os.getenv("MGAP_DB_URL"):
            s.db_url = v
        if v := os.getenv("MGAP_REGISTRY_PATH"):
            s.registry_path = Path(v)
        if v := os.getenv("MGAP_ARTIFACTS_DIR"):
            s.artifacts_dir = Path(v)
        if v := os.getenv("MGAP_RESULTS_DIR"):
            s.results_dir = Path(v)
        if v := os.getenv("MGAP_USE_LLM"):
            s.use_llm = v.lower() not in ("0", "false", "no")
        if v := os.getenv("MGAP_DEFAULT_TOP_K"):
            try:
                s.default_top_k = int(v)
            except ValueError:
                pass
        if v := os.getenv("MGAP_MIN_RESONANCE"):
            try:
                s.min_resonance = float(v)
            except ValueError:
                pass
        if v := os.getenv("MGAP_LOG_LEVEL"):
            s.log_level = v.upper()
        if v := os.getenv("MGAP_GAP_MODE"):
            if v in ("max", "mean", "rms"):
                s.gap_mode = v
        return s

    def ensure_dirs(self):
        """Создаёт директории если не существуют."""
        self.artifacts_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)


# Глобальный синглтон
settings = MGAPSettings.from_env()
