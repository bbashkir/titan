# -*- coding: utf-8 -*-
"""
Подготовка признаков и оценка соответствия сплава запросу пользователя.

Модуль общий для обеих моделей (поиска похожих и генеративной):
  * превращает записи БД в таблицу pandas и векторы состава;
  * хранит структуру пользовательского запроса (UserQuery);
  * считает процент соответствия сплава запросу и формирует
    текстовые объяснения «соответствует / не соответствует».
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from database.models import Alloy, get_all_alloys
from logger_setup import get_logger

log = get_logger("ml.features")

# Порядковая шкала коррозионной стойкости
CORROSION_ORDER = {"low": 0, "medium": 1, "high": 2}
CORROSION_LABELS_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}

# Синонимы областей применения: канонический термин -> ключевые слова
# (русские и английские). Используются для сопоставления текстового запроса
# с полем application записи.
APPLICATION_SYNONYMS: Dict[str, List[str]] = {
    "авиация": [
        "авиа", "самолёт", "самолет", "планер", "лопатк", "двигател", "гтд",
        "аэрокосм", "космос", "ракет", "aerospace", "aircraft", "aeroengine",
        "turbine", "airframe", "blade", "disc", "compressor", "landing gear",
    ],
    "медицина": [
        "медиц", "имплант", "протез", "эндопротез", "кост", "стоматолог",
        "хирург", "biomedical", "implant", "orthop", "dental", "surgical",
        "prosthes", "biocompatib", "bone",
    ],
    "судостроение": [
        "судострое", "морск", "корабл", "подводн", "marine", "naval",
        "seawater", "submarine", "offshore", "ship",
    ],
    "химическая промышленность": [
        "химическ", "агрессивн", "теплообмен", "реактор", "трубопровод",
        "chemical", "corrosive", "heat exchanger", "desalination",
        "pressure vessel", "autoclave",
    ],
    "энергетика": [
        "энергетик", "атомн", "аэс", "конденсатор", "турбин",
        "power", "nuclear", "energy", "condenser",
    ],
    "криогеника": [
        "криоген", "низких температур", "низкотемператур",
        "cryogenic", "low temperature",
    ],
    "спорт и товары народного потребления": [
        "спорт", "велосипед", "sports", "bicycle", "consumer",
    ],
    "крепёж и метизы": [
        "крепёж", "крепеж", "болт", "винт", "заклёпк", "заклепк",
        "fastener", "bolt", "screw", "rivet", "spring",
    ],
}


# ==========================================================================
# ПОЛЬЗОВАТЕЛЬСКИЙ ЗАПРОС
# ==========================================================================

@dataclass
class UserQuery:
    """
    Требования пользователя к сплаву.

    ranges       — числовые свойства: имя поля -> (минимум, максимум).
                   Любая из границ может быть None (ограничение односторонее).
    corrosion    — требуемый уровень коррозионной стойкости (low/medium/high).
    application  — область применения, введённая текстом.
    """

    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    corrosion: Optional[str] = None
    application: Optional[str] = None

    def is_empty(self) -> bool:
        """Пустой ли запрос (не задано ни одного критерия)."""
        return not self.ranges and not self.corrosion and not self.application

    def criteria_count(self) -> int:
        """Сколько критериев задал пользователь."""
        return len(self.ranges) + int(bool(self.corrosion)) + int(bool(self.application))

    def describe(self) -> List[str]:
        """Человекочитаемое описание запроса — для вывода и PDF-отчёта."""
        lines: List[str] = []
        for prop, (vmin, vmax) in self.ranges.items():
            label, unit = config.PROPERTY_LABELS[prop]
            if vmin is not None and vmax is not None:
                lines.append(f"{label}: от {fmt(vmin)} до {fmt(vmax)} {unit}".strip())
            elif vmin is not None:
                lines.append(f"{label}: не менее {fmt(vmin)} {unit}".strip())
            elif vmax is not None:
                lines.append(f"{label}: не более {fmt(vmax)} {unit}".strip())
        if self.corrosion:
            lines.append(
                f"Коррозионная стойкость: не ниже уровня "
                f"«{CORROSION_LABELS_RU.get(self.corrosion, self.corrosion)}»"
            )
        if self.application:
            lines.append(f"Область применения: {self.application}")
        return lines


@dataclass
class MatchResult:
    """Результат оценки одного сплава по запросу пользователя."""

    percent: float                       # итоговое соответствие, 0…100 %
    matched: List[str] = field(default_factory=list)    # что соответствует
    unmatched: List[str] = field(default_factory=list)  # что не соответствует
    unknown: List[str] = field(default_factory=list)    # по чему нет данных


# ==========================================================================
# ФОРМАТИРОВАНИЕ
# ==========================================================================

def fmt(value: Optional[float], digits: int = 2) -> str:
    """
    Форматирует число для вывода. None превращается в прочерк «—»,
    целые значения печатаются без дробной части.
    """
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def format_composition(comp: Optional[Dict[str, float]]) -> str:
    """
    Строка химического состава вида «Ti: 89.4%, Al: 6.0%, V: 4.2%».
    Титан выводится первым, остальные элементы — по убыванию доли.
    """
    if not comp:
        return "—"
    items = sorted(
        comp.items(),
        key=lambda kv: (kv[0] != "Ti", -kv[1]),
    )
    return ", ".join(f"{element}: {value:.1f}%" for element, value in items)


# ==========================================================================
# ПРЕОБРАЗОВАНИЕ БАЗЫ В ТАБЛИЦУ
# ==========================================================================

def composition_to_vector(comp: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Вектор состава фиксированной длины по списку config.ELEMENTS, в процентах.
    Элементы, отсутствующие в списке, суммируются в долю титана
    (они всё равно являются частью основы).
    """
    vector = np.zeros(len(config.ELEMENTS), dtype=float)
    if not comp:
        return vector
    index = {element: i for i, element in enumerate(config.ELEMENTS)}
    extra = 0.0
    for element, value in comp.items():
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if element in index:
            vector[index[element]] += amount
        else:
            extra += amount
    if extra and "Ti" in index:
        vector[index["Ti"]] += extra
    return vector


def vector_to_composition(vector: np.ndarray,
                          min_fraction: float = config.GEN_MIN_ELEMENT) -> Dict[str, float]:
    """
    Обратное преобразование: вектор -> словарь состава.
    Элементы с долей меньше min_fraction отбрасываются, остаток
    перенормируется до ровно 100 %.
    """
    raw = {
        element: float(value)
        for element, value in zip(config.ELEMENTS, vector)
        if value >= min_fraction
    }
    if not raw:
        return {}
    total = sum(raw.values())
    normalized = {element: value * 100.0 / total for element, value in raw.items()}

    # Округляем до 0.1 % и «сбрасываем» ошибку округления на титан,
    # чтобы сумма гарантированно равнялась 100.0 %.
    rounded = {element: round(value, 1) for element, value in normalized.items()}
    residual = round(100.0 - sum(rounded.values()), 1)
    if "Ti" in rounded:
        rounded["Ti"] = round(rounded["Ti"] + residual, 1)
    else:
        heaviest = max(rounded, key=rounded.get)
        rounded[heaviest] = round(rounded[heaviest] + residual, 1)
    return rounded


def build_dataframe(session) -> pd.DataFrame:
    """
    Загружает всю базу в pandas.DataFrame.

    Колонки:
        id, name, source, corrosion_resistance, application,
        <числовые свойства>, el_<символ элемента> для каждого элемента,
        composition (исходный словарь).
    """
    alloys: List[Alloy] = get_all_alloys(session)
    rows = []
    for alloy in alloys:
        comp = alloy.composition
        row = {
            "id": alloy.id,
            "name": alloy.name,
            "source": alloy.source,
            "corrosion_resistance": alloy.corrosion_resistance,
            "application": alloy.application or "",
            "composition": comp,
        }
        row.update(alloy.properties_dict())
        for element, value in zip(config.ELEMENTS, composition_to_vector(comp)):
            row[f"el_{element}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("База данных пуста — модели обучить невозможно")
    else:
        log.info("Загружено записей для обучения: %d", len(df))
    return df


def element_matrix(df: pd.DataFrame) -> np.ndarray:
    """Матрица составов (строки — сплавы, столбцы — элементы), в процентах."""
    columns = [f"el_{element}" for element in config.ELEMENTS]
    return df[columns].to_numpy(dtype=float)


# ==========================================================================
# ОЦЕНКА СООТВЕТСТВИЯ ЗАПРОСУ
# ==========================================================================

def _range_span(vmin: Optional[float], vmax: Optional[float]) -> float:
    """
    Характерный масштаб запрошенного диапазона.
    Нужен, чтобы отклонение оценивалось относительно, а не абсолютно:
    отклонение в 50 МПа для прочности и для модуля упругости — разные вещи.
    """
    if vmin is not None and vmax is not None:
        return max(vmax - vmin, abs(vmax) * 0.05, 1e-6)
    bound = vmin if vmin is not None else vmax
    return max(abs(bound) * 0.25, 1e-6)


def score_numeric(prop: str, value: Optional[float],
                  vmin: Optional[float], vmax: Optional[float]) -> Tuple[Optional[float], str]:
    """
    Оценивает одно числовое свойство.

    Возвращает пару (оценка 0…1 или None, текстовое пояснение).
    None означает «в источнике нет данных» — такие параметры
    учитываются как половинное соответствие, но выносятся в отдельный список.
    """
    label, unit = config.PROPERTY_LABELS[prop]

    if value is None:
        return None, f"{label}: данные отсутствуют (—)"

    # Формулировка запрошенного условия
    if vmin is not None and vmax is not None:
        requested = f"запрошено {fmt(vmin)}–{fmt(vmax)} {unit}".strip()
    elif vmin is not None:
        requested = f"запрошено от {fmt(vmin)} {unit}".strip()
    else:
        requested = f"запрошено до {fmt(vmax)} {unit}".strip()

    actual = f"{fmt(value)} {unit}".strip()

    # Попадание в диапазон
    inside = (vmin is None or value >= vmin) and (vmax is None or value <= vmax)
    if inside:
        return 1.0, f"{label} {actual} ({requested})"

    # Отклонение от ближайшей границы
    if vmin is not None and value < vmin:
        deviation = vmin - value
        direction = "ниже"
        bound = vmin
    else:
        deviation = value - vmax
        direction = "выше"
        bound = vmax

    span = _range_span(vmin, vmax)
    max_deviation = span * 2.0 * config.TOLERANCE_RATIO
    score = max(0.0, 1.0 - deviation / max_deviation)

    explanation = (
        f"{label} {actual} — на {fmt(deviation)} {unit} {direction} "
        f"запрошенных {fmt(bound)} {unit}".strip()
    )
    return score, explanation


def score_corrosion(value: Optional[str], requested: str) -> Tuple[Optional[float], str]:
    """Оценивает коррозионную стойкость по порядковой шкале low < medium < high."""
    requested_ru = CORROSION_LABELS_RU.get(requested, requested)
    if not value or value not in CORROSION_ORDER:
        return None, "Коррозионная стойкость: данные отсутствуют (—)"

    actual_ru = CORROSION_LABELS_RU.get(value, value)
    difference = CORROSION_ORDER[requested] - CORROSION_ORDER[value]

    if difference <= 0:
        # Стойкость не ниже требуемой — условие выполнено
        return 1.0, f"Коррозионная стойкость: {actual_ru} (запрошено не ниже «{requested_ru}»)"

    score = max(0.0, 1.0 - 0.5 * difference)
    return score, (
        f"Коррозионная стойкость: {actual_ru} — ниже запрошенного "
        f"уровня «{requested_ru}»"
    )


def _application_keywords(text: str) -> List[str]:
    """
    Подбирает ключевые слова для поиска по тексту области применения.

    Если запрос попадает в одну из известных категорий («авиация», «медицина»
    и т. д.), берутся все её синонимы — так запрос «авиация» найдёт запись
    с текстом «aerospace, turbine blades». Иначе используются сами слова
    запроса.
    """
    lowered = text.lower()
    keywords: List[str] = []
    for canonical, synonyms in APPLICATION_SYNONYMS.items():
        if canonical in lowered or any(syn in lowered for syn in synonyms):
            keywords.extend(synonyms)
            keywords.append(canonical)
    if not keywords:
        # Явной категории нет — используем сами слова запроса длиной от 4 букв
        keywords = [word.strip(".,;:()") for word in lowered.split() if len(word) >= 4]
    return keywords


def score_application(value: Optional[str], requested: str) -> Tuple[Optional[float], str]:
    """Оценивает соответствие области применения текстовому запросу."""
    if not value:
        return None, f"Область применения: данные отсутствуют (—), запрошено «{requested}»"

    lowered = value.lower()
    keywords = _application_keywords(requested)
    hits = [kw for kw in keywords if kw and kw in lowered]

    if hits:
        return 1.0, f"Область применения соответствует запросу «{requested}»"
    return 0.2, (
        f"Область применения «{_shorten(value)}» напрямую не связана "
        f"с запросом «{requested}»"
    )


def _shorten(text: str, limit: int = 70) -> str:
    """Обрезает длинный текст для компактного вывода."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def evaluate_match(properties: Dict[str, Optional[float]],
                   corrosion: Optional[str],
                   application: Optional[str],
                   query: UserQuery) -> MatchResult:
    """
    Считает итоговый процент соответствия сплава запросу.

    Каждый заданный пользователем критерий даёт оценку от 0 до 1;
    итог — среднее арифметическое, переведённое в проценты.
    Критерии, по которым в источнике нет данных, засчитываются как 0.5
    (нейтрально) и перечисляются отдельно, чтобы пользователь видел,
    что оценка неполная.
    """
    if query.is_empty():
        return MatchResult(percent=0.0)

    scores: List[float] = []
    matched: List[str] = []
    unmatched: List[str] = []
    unknown: List[str] = []

    for prop, (vmin, vmax) in query.ranges.items():
        score, explanation = score_numeric(prop, properties.get(prop), vmin, vmax)
        if score is None:
            unknown.append(explanation)
            scores.append(0.5)
        elif score >= 1.0:
            matched.append(explanation)
            scores.append(1.0)
        else:
            unmatched.append(explanation)
            scores.append(score)

    if query.corrosion:
        score, explanation = score_corrosion(corrosion, query.corrosion)
        if score is None:
            unknown.append(explanation)
            scores.append(0.5)
        elif score >= 1.0:
            matched.append(explanation)
            scores.append(1.0)
        else:
            unmatched.append(explanation)
            scores.append(score)

    if query.application:
        score, explanation = score_application(application, query.application)
        if score is None:
            unknown.append(explanation)
            scores.append(0.5)
        elif score >= 1.0:
            matched.append(explanation)
            scores.append(1.0)
        else:
            unmatched.append(explanation)
            scores.append(score)

    percent = 100.0 * float(np.mean(scores)) if scores else 0.0
    return MatchResult(
        percent=round(percent, 1),
        matched=matched,
        unmatched=unmatched,
        unknown=unknown,
    )


def build_explanation(result: MatchResult) -> str:
    """
    Собирает связное объяснение из результата оценки:
    «Соответствует запросу по: … Не соответствует: … Нет данных: …».
    """
    parts: List[str] = []
    if result.matched:
        parts.append("Соответствует запросу по: " + "; ".join(result.matched) + ".")
    if result.unmatched:
        parts.append("Не соответствует: " + "; ".join(result.unmatched) + ".")
    if result.unknown:
        parts.append("Нет данных по: " + "; ".join(result.unknown) + ".")
    return " ".join(parts) if parts else "Критерии не заданы."
