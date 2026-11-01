# -*- coding: utf-8 -*-
"""
Модели базы данных и функции работы с ней.

База — SQLite, доступ через SQLAlchemy ORM.
Основная таблица — alloys: одна запись = один сплав из одного источника.
"""

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

import config
from logger_setup import get_logger

log = get_logger("database.models")

Base = declarative_base()


class Alloy(Base):
    """
    Запись о титановом сплаве, полученная из одного конкретного источника.

    Один и тот же сплав, найденный в разных источниках, хранится
    несколькими записями: name совпадает, source различается.
    Незаполненные параметры хранятся как NULL и выводятся прочерком.
    """

    __tablename__ = "alloys"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Название сплава: ВТ6, Ti-6Al-4V, Grade 5 и т.п.
    name = Column(String(120), nullable=False, index=True)

    # Источник: URL страницы или обозначение стандарта (например «ГОСТ 19807-91»)
    source = Column(String(500), nullable=False)

    # Химический состав в виде JSON-строки: {"Ti": 90.0, "Al": 6.0, "V": 4.0}
    # Значения — массовые доли в процентах.
    chemical_composition = Column(Text, nullable=True)

    # Хеш нормализованного состава — служебное поле для быстрой дедупликации
    composition_hash = Column(String(64), nullable=True, index=True)

    density = Column(Float, nullable=True)              # г/см³
    melting_point = Column(Float, nullable=True)        # °C
    tensile_strength = Column(Float, nullable=True)     # МПа
    yield_strength = Column(Float, nullable=True)       # МПа
    elastic_modulus = Column(Float, nullable=True)      # ГПа
    hardness_brinell = Column(Float, nullable=True)     # HB
    corrosion_resistance = Column(String(10), nullable=True)  # low / medium / high
    thermal_conductivity = Column(Float, nullable=True) # Вт/(м·К)
    fatigue_strength = Column(Float, nullable=True)     # МПа
    application = Column(Text, nullable=True)           # область применения

    # Дата и время получения записи
    parsed_at = Column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        # Правило дедупликации: name + source + состав уникальны в совокупности
        UniqueConstraint(
            "name", "source", "composition_hash", name="uq_alloy_name_source_comp"
        ),
        Index("ix_alloy_name_source", "name", "source"),
    )

    # ---------------------------------------------------------------- утилиты

    @property
    def composition(self) -> Dict[str, float]:
        """Химический состав как словарь Python (пустой словарь, если данных нет)."""
        if not self.chemical_composition:
            return {}
        try:
            return json.loads(self.chemical_composition)
        except json.JSONDecodeError:
            log.warning("Некорректный JSON состава у записи id=%s", self.id)
            return {}

    def properties_dict(self) -> Dict[str, Optional[float]]:
        """Все числовые свойства записи одним словарём."""
        return {prop: getattr(self, prop) for prop in config.NUMERIC_PROPERTIES}

    def __repr__(self) -> str:
        return f"<Alloy id={self.id} name={self.name!r} source={self.source[:40]!r}>"


# --------------------------------------------------------------------------
# ПОДКЛЮЧЕНИЕ К БД
# --------------------------------------------------------------------------

# echo=False — не засорять лог SQL-запросами.
# check_same_thread=False нужен, т.к. планировщик работает в отдельном потоке.
engine = create_engine(
    config.DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)

SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Создаёт таблицы, если их ещё нет. Безопасно вызывать многократно."""
    Base.metadata.create_all(engine)
    log.info("База данных инициализирована: %s", config.DB_PATH)


@contextmanager
def session_scope():
    """
    Контекстный менеджер сессии: автоматический commit при успехе
    и rollback при исключении.
    """
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# --------------------------------------------------------------------------

def normalize_composition(comp: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Приводит состав к каноническому виду: элементы отсортированы по алфавиту,
    доли округлены до двух знаков. Нужно для устойчивого сравнения и хеширования.
    """
    if not comp:
        return None
    cleaned = {}
    for element, value in comp.items():
        if value is None:
            continue
        try:
            cleaned[str(element).strip().capitalize()] = round(float(value), 2)
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return None
    return {k: cleaned[k] for k in sorted(cleaned)}


def composition_hash(comp: Optional[Dict[str, float]]) -> str:
    """
    SHA-256 от канонического представления состава.
    Для пустого состава возвращает хеш строки 'null', чтобы поле не было NULL
    (иначе UNIQUE-ограничение в SQLite перестаёт работать: NULL != NULL).
    """
    normalized = normalize_composition(comp)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_alloy(session, record: Dict) -> str:
    """
    Добавляет запись о сплаве, если такой ещё нет.

    Возвращает одну из строк:
        "added"     — запись добавлена как новая;
        "variant"   — сплав из этого источника уже был, но состав отличается,
                      поэтому добавлена ещё одна запись (новый вариант данных);
        "duplicate" — полное совпадение name + source + состав, запись пропущена.

    Старые записи никогда не изменяются и не удаляются — это требование
    к режиму автообновления.
    """
    name = (record.get("name") or "").strip()
    source = (record.get("source") or "").strip()
    if not name or not source:
        log.warning("Пропущена запись без имени или источника: %s", record)
        return "duplicate"

    comp = normalize_composition(record.get("chemical_composition"))
    comp_hash = composition_hash(comp)

    # 1. Полный дубль: то же имя, тот же источник, тот же состав
    exists_exact = (
        session.query(Alloy.id)
        .filter(
            Alloy.name == name,
            Alloy.source == source,
            Alloy.composition_hash == comp_hash,
        )
        .first()
    )
    if exists_exact:
        return "duplicate"

    # 2. Пара name + source уже встречалась, но состав другой —
    #    сохраняем как отдельный вариант и отмечаем это в логе.
    exists_pair = (
        session.query(Alloy.id)
        .filter(Alloy.name == name, Alloy.source == source)
        .first()
    )
    status = "variant" if exists_pair else "added"

    alloy = Alloy(
        name=name,
        source=source,
        chemical_composition=(
            json.dumps(comp, ensure_ascii=False) if comp else None
        ),
        composition_hash=comp_hash,
        density=record.get("density"),
        melting_point=record.get("melting_point"),
        tensile_strength=record.get("tensile_strength"),
        yield_strength=record.get("yield_strength"),
        elastic_modulus=record.get("elastic_modulus"),
        hardness_brinell=record.get("hardness_brinell"),
        corrosion_resistance=record.get("corrosion_resistance"),
        thermal_conductivity=record.get("thermal_conductivity"),
        fatigue_strength=record.get("fatigue_strength"),
        application=record.get("application"),
        parsed_at=record.get("parsed_at") or datetime.now(),
    )
    session.add(alloy)
    session.flush()  # получаем id и ловим нарушение ограничений сразу
    return status


def get_all_alloys(session) -> List[Alloy]:
    """
    Все записи БД, сгруппированные по названию сплава:
    записи с одинаковым name идут подряд, внутри группы — по источнику.
    """
    return (
        session.query(Alloy)
        .order_by(Alloy.name.asc(), Alloy.source.asc(), Alloy.id.asc())
        .all()
    )


def get_stats(session) -> Dict[str, int]:
    """Краткая статистика по базе: сколько записей, уникальных сплавов, источников."""
    total = session.query(func.count(Alloy.id)).scalar() or 0
    unique_names = session.query(func.count(func.distinct(Alloy.name))).scalar() or 0
    unique_sources = session.query(func.count(func.distinct(Alloy.source))).scalar() or 0
    last_parsed = session.query(func.max(Alloy.parsed_at)).scalar()
    return {
        "total": total,
        "unique_names": unique_names,
        "unique_sources": unique_sources,
        "last_parsed": last_parsed,
    }


def group_by_name(alloys: Iterable[Alloy]) -> List[Tuple[str, List[Alloy]]]:
    """
    Группирует список записей по названию сплава.
    Возвращает список пар (название, [записи]) в алфавитном порядке.
    """
    grouped: Dict[str, List[Alloy]] = {}
    for alloy in alloys:
        grouped.setdefault(alloy.name, []).append(alloy)
    return sorted(grouped.items(), key=lambda item: item[0])
