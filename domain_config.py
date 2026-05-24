# domain_config.py — HX-AM v4.6
"""
Расширенный список доменов с полной совместимостью:
- DOMAIN_PARAMS для FourDBuilder
- DOMAIN_ALIASES для нормализации  
- DOMAIN_UNESCO для MGAP-маппинга
- DOMAIN_KEYWORDS для быстрого определения без LLM
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# ── Полный список валидных доменов ────────────────────────────────────────────
VALID_DOMAINS: List[str] = [
    # Исходные 15
    "biology", "chemistry", "physics", "economics", "psychology",
    "linguistics", "sociology", "geology", "ecology", "neuroscience",
    "medicine", "astronomy", "history", "architecture", "general",
    # Новые (7 — только с реальной потребностью по логам)
    "materials_science",   # материаловедение, память формы
    "computer_science",    # CS, алгоритмы, сети
    "cognitive_science",   # когнитивистика (отдельно от psychology)
    "engineering",         # инженерия, механика систем
    "political_science",   # политология (уже в domain_map.json MGAP)
    "anthropology",        # антропология, культура (→ sociology-смежный)
    "philosophy",          # философия, эпистемология
]

# ── domain_distance через эмбеддинги (кэш центроидов) ─────────────────────────
# При маленьком архиве <15 артефактов в домене — используем
# статическое семантическое расстояние между доменами.
# Значения: 0.0 = идентичные, 1.0 = максимально далёкие.
DOMAIN_STATIC_DISTANCE: Dict[Tuple[str,str], float] = {
    ("biology","medicine"): 0.25,
    ("biology","ecology"): 0.30,
    ("biology","chemistry"): 0.45,
    ("biology","neuroscience"): 0.40,
    ("neuroscience","psychology"): 0.30,
    ("neuroscience","cognitive_science"): 0.25,
    ("psychology","cognitive_science"): 0.20,
    ("psychology","sociology"): 0.40,
    ("sociology","anthropology"): 0.25,
    ("sociology","political_science"): 0.35,
    ("economics","political_science"): 0.38,
    ("economics","sociology"): 0.42,
    ("physics","materials_science"): 0.30,
    ("physics","engineering"): 0.35,
    ("physics","chemistry"): 0.38,
    ("materials_science","engineering"): 0.25,
    ("computer_science","mathematics"): 0.30,
    ("computer_science","engineering"): 0.38,
    ("linguistics","psychology"): 0.45,
    ("linguistics","cognitive_science"): 0.40,
    ("philosophy","cognitive_science"): 0.42,
    ("philosophy","history"): 0.38,
}

# ── Параметры 4D для новых доменов ───────────────────────────────────────────
NEW_DOMAIN_PARAMS: Dict[str, Dict] = {
    "materials_science": {
        "k": (3, 8),    "C": (0.38, 0.68), "D": (1.6, 2.8),
        "tau": (0.05, 1.5), "H": (0.45, 0.70), "freq": (1.0, 8.0),
        "h": (0.5, 3.0),  "T": (0.6, 2.5),  "eta": (0.08, 0.28),
    },
    "computer_science": {
        "k": (8, 30),   "C": (0.40, 0.72), "D": (2.0, 3.2),
        "tau": (0.01, 0.5), "H": (0.40, 0.65), "freq": (2.0, 10.0),
        "h": (0.5, 2.5),  "T": (0.8, 1.8),  "eta": (0.10, 0.30),
    },
    "cognitive_science": {
        "k": (8, 18),   "C": (0.55, 0.82), "D": (2.0, 2.7),
        "tau": (0.2, 2.0),  "H": (0.55, 0.78), "freq": (1.0, 6.0),
        "h": (0.8, 2.5),  "T": (0.9, 1.8),  "eta": (0.12, 0.32),
    },
    "engineering": {
        "k": (5, 15),   "C": (0.42, 0.72), "D": (1.8, 2.6),
        "tau": (0.1, 3.0),  "H": (0.48, 0.70), "freq": (0.5, 5.0),
        "h": (0.8, 2.8),  "T": (0.8, 2.0),  "eta": (0.12, 0.32),
    },
    "political_science": {
        "k": (10, 25),  "C": (0.45, 0.72), "D": (2.0, 2.8),
        "tau": (1.0, 6.0),  "H": (0.52, 0.74), "freq": (0.2, 1.5),
        "h": (1.5, 3.5),  "T": (1.0, 2.2),  "eta": (0.25, 0.50),
    },
    "anthropology": {
        "k": (10, 22),  "C": (0.50, 0.76), "D": (2.0, 2.7),
        "tau": (2.0, 10.0), "H": (0.58, 0.78), "freq": (0.1, 1.0),
        "h": (1.0, 2.8),  "T": (1.0, 2.0),  "eta": (0.20, 0.42),
    },
    "philosophy": {
        "k": (6, 16),   "C": (0.48, 0.78), "D": (1.8, 2.6),
        "tau": (1.0, 8.0),  "H": (0.55, 0.80), "freq": (0.2, 2.0),
        "h": (0.5, 2.0),  "T": (0.8, 1.8),  "eta": (0.15, 0.38),
    },
}

# ── Ключевые слова для быстрого определения домена (без LLM) ────────────────
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "materials_science": [
        "материаловедение", "сплав", "кристалл", "память формы", "нанотрубк",
        "полимер", "магнитн", "ферромагнит", "металл", "керамик",
        "materials", "alloy", "crystal", "shape memory", "polymer", "lattice",
    ],
    "computer_science": [
        "алгоритм", "граф алгоритм", "сеть нейрон", "машинное обучение",
        "распределённ", "база данных", "кибербезопасн", "шифрован",
        "algorithm", "machine learning", "neural network", "distributed",
        "compression", "complexity",
    ],
    "cognitive_science": [
        "когнитивн", "восприяти", "внимани", "рабочая память", "прайминг",
        "когнитивная нагрузка", "схема", "прототип", "когнитивное",
        "perception", "attention", "working memory", "priming", "schema",
        "cognitive load", "categorization",
    ],
    "engineering": [
        "инженерн", "механическ", "конструкци", "надёжность", "отказ",
        "гидравлик", "термодинамик системы", "управлени систем",
        "engineering", "mechanical", "structural", "reliability", "failure mode",
        "hydraulic", "control system", "feedback control",
    ],
    "political_science": [
        "политическ", "власть", "государств", "демократи", "режим",
        "коалици", "партий", "электорал", "геополитик",
        "political", "power", "state", "democracy", "regime", "coalition",
        "electoral", "governance",
    ],
    "anthropology": [
        "культур", "антропологи", "обряд", "ритуал", "традици",
        "социальная структура", "обмен", "трибальн", "этнограф",
        "culture", "ritual", "tradition", "kinship", "ethnography",
        "symbolic", "exchange",
    ],
    "philosophy": [
        "философи", "эпистемологи", "онтологи", "этик", "метафизик",
        "феноменологи", "диалектик", "логик", "истина", "сознание",
        "philosophy", "epistemology", "ontology", "ethics", "metaphysics",
        "phenomenology", "dialectic", "consciousness",
    ],
    # Усиление существующих — часто попадают в general
    "sociology": [
        "социальн", "коллективн поведени", "норм", "статус", "группов",
        "коллектив", "институт", "страти", "социальн капитал",
    ],
    "psychology": [
        "психологи", "поведени", "личность", "мотивац", "эмоци",
        "бихевиорист", "психотерапи", "стресс", "адаптаци",
    ],
    "linguistics": [
        "язык", "семантик", "синтаксис", "морфологи", "прагматик",
        "речь", "текст", "дискурс", "лингвистик",
    ],
}

# ── Нормализация: псевдонимы → канонический домен ────────────────────────────
DOMAIN_ALIASES: Dict[str, str] = {
    # Русские названия
    "биология": "biology", "химия": "chemistry", "физика": "physics",
    "экономика": "economics", "психология": "psychology",
    "лингвистика": "linguistics", "социология": "sociology",
    "геология": "geology", "экология": "ecology",
    "нейронаука": "neuroscience", "медицина": "medicine",
    "астрономия": "astronomy", "история": "history",
    "архитектура": "architecture", "общий": "general",
    "материаловедение": "materials_science",
    "информатика": "computer_science",
    "когнитивистика": "cognitive_science",
    "инженерия": "engineering",
    "политология": "political_science",
    "антропология": "anthropology",
    "философия": "philosophy",
    "математика": "mathematics",
    # Английские варианты
    "math": "mathematics", "bio": "biology", "chem": "chemistry",
    "phys": "physics", "econ": "economics", "psych": "psychology",
    "social": "sociology", "neuro": "neuroscience",
    "geo": "geology", "ling": "linguistics",
    "cs": "computer_science", "ai": "computer_science",
    "polisci": "political_science", "phil": "philosophy",
    "anthro": "anthropology", "cogni": "cognitive_science",
}

# ── UNESCO-маппинг для MGAP ───────────────────────────────────────────────────
DOMAIN_UNESCO: Dict[str, Dict] = {
    "materials_science": {"disc_code": "2", "sector_code": "2.3"},
    "computer_science":  {"disc_code": "2", "sector_code": "2.4"},
    "cognitive_science": {"disc_code": "6", "sector_code": "6.1"},
    "engineering":       {"disc_code": "2", "sector_code": "2.2"},
    "political_science": {"disc_code": "5", "sector_code": "5.4"},
    "anthropology":      {"disc_code": "5", "sector_code": "5.2"},
    "philosophy":        {"disc_code": "6", "sector_code": "6.3"},
}