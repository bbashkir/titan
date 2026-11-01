# -*- coding: utf-8 -*-
"""
Самопроверка системы: сквозной прогон всех этапов работы.

Запуск:
    python main.py --selftest
    docker compose run --rm app --selftest

Проверяются ключевые требования к системе:
  1. парсинг наполняет базу;
  2. повторный парсинг не создаёт дублей;
  3. один сплав из разных источников хранится отдельными записями;
  4. незаполненные параметры хранятся как NULL и выводятся прочерком;
  5. обе модели обучаются на данных базы;
  6. блок А возвращает существующие сплавы с процентом соответствия;
  7. блок Б возвращает составы, которых нет в базе, с суммой элементов
     ровно 100 % и содержанием титана не ниже 50 %;
  8. отчёт выгружается в PDF.

Проверка не разрушающая: она только добавляет записи (которых нет)
и создаёт один PDF-файл.
"""

from datetime import datetime
from typing import List

import config
from database.models import (
    get_all_alloys,
    get_stats,
    group_by_name,
    init_db,
    session_scope,
)
from ml.features import UserQuery, build_dataframe, format_composition
from ml.generative_model import GenerativeModel
from ml.similarity_model import SimilarityModel
from parser.scraper import run_parsing
from reports.pdf_generator import generate_pdf_report

# Накопитель результатов: (пройдена, описание)
_results: List[tuple] = []


def check(condition: bool, description: str) -> bool:
    """Фиксирует результат одной проверки и печатает его."""
    mark = "OK  " if condition else "СБОЙ"
    print(f"  [{mark}] {description}")
    _results.append((bool(condition), description))
    return bool(condition)


def run() -> int:
    """
    Выполняет полный прогон.
    Возвращает код возврата процесса: 0 — всё в порядке, 1 — есть сбои.
    """
    print("=" * 70)
    print("САМОПРОВЕРКА СИСТЕМЫ ПОДБОРА ТИТАНОВЫХ СПЛАВОВ")
    print("=" * 70)

    init_db()

    # --- 1. Парсинг и дедупликация ------------------------------------
    print("\n1. Парсинг и дедупликация")
    first = run_parsing(["reference"])
    second = run_parsing(["reference"])

    check(first["collected"] > 0, f"источник вернул записи: {first['collected']}")
    check(
        second["added"] == 0,
        f"повторный парсинг не добавил новых записей "
        f"(добавлено: {second['added']}, пропущено дублей: {second['duplicate']})",
    )
    check(
        second["duplicate"] == second["collected"],
        "все записи повторного парсинга опознаны как дубли",
    )

    # --- 2. Структура базы --------------------------------------------
    print("\n2. Структура и содержимое базы")
    with session_scope() as session:
        stats = get_stats(session)
        alloys = get_all_alloys(session)
        groups = group_by_name(alloys)

        check(stats["total"] > 0, f"записей в базе: {stats['total']}")
        check(
            stats["unique_sources"] > 1,
            f"использовано источников: {stats['unique_sources']}",
        )

        multi_source = [(name, len(rows)) for name, rows in groups if len(rows) > 1]
        check(
            bool(multi_source),
            f"один сплав из разных источников хранится отдельно: "
            f"{', '.join(f'{n} ({c})' for n, c in multi_source[:4])}",
        )

        # Группировка: записи с одинаковым name должны идти подряд
        names_in_order = [alloy.name for alloy in alloys]
        grouped_properly = all(
            names_in_order.index(name) + count - 1
            == len(names_in_order) - 1 - names_in_order[::-1].index(name)
            for name, count in ((n, len(r)) for n, r in groups)
        )
        check(grouped_properly, "записи сгруппированы по названию сплава")

        # NULL для ненайденных параметров
        with_nulls = [
            alloy for alloy in alloys
            if any(getattr(alloy, prop) is None for prop in config.NUMERIC_PROPERTIES)
        ]
        check(
            bool(with_nulls),
            f"параметры без данных хранятся как NULL "
            f"(записей с пропусками: {len(with_nulls)})",
        )

    # --- 3. Обучение моделей ------------------------------------------
    print("\n3. Обучение моделей")
    with session_scope() as session:
        df = build_dataframe(session)

    similarity = SimilarityModel().fit(df)
    check(similarity.is_fitted, "модель поиска похожих обучена (kNN + Random Forest)")
    check(
        len(similarity.predictor.models) > 0,
        f"обучены регрессоры свойств: {len(similarity.predictor.models)} шт.",
    )

    generative = GenerativeModel(predictor=similarity.predictor).fit(df)
    check(generative.is_fitted, "генеративная модель обучена (VAE)")

    generative.save()
    check(generative.load(), "веса VAE сохраняются и читаются обратно")

    # --- 4. Прогнозирование -------------------------------------------
    print("\n4. Прогнозирование по запросу")
    query = UserQuery(
        ranges={"tensile_strength": (900.0, None), "density": (None, 4.6)},
        corrosion="high",
        application="авиация",
    )
    print("   Запрос: " + "; ".join(query.describe()))

    similar = similarity.find_similar(query, top_n=config.TOP_SIMILAR)
    check(
        len(similar) == config.TOP_SIMILAR,
        f"блок А содержит {len(similar)} существующих сплавов",
    )
    check(
        all(0.0 <= item.percent <= 100.0 for item in similar),
        "процент соответствия в блоке А лежит в диапазоне 0…100",
    )
    check(
        all(item.explanation for item in similar),
        "для каждого сплава блока А сформировано объяснение",
    )
    check(
        similar == sorted(similar, key=lambda i: -i.percent),
        "результаты блока А отсортированы по убыванию соответствия",
    )
    for item in similar[:3]:
        print(f"      {item.name} — {item.percent:.1f} % — "
              f"{format_composition(item.composition)}")

    generated = generative.generate(
        query, top_n=config.TOP_GENERATED, similarity_model=similarity
    )
    check(
        len(generated) == config.TOP_GENERATED,
        f"блок Б содержит {len(generated)} сгенерированных составов",
    )

    sums_ok = all(abs(sum(item.composition.values()) - 100.0) < 0.05 for item in generated)
    check(sums_ok, "сумма долей элементов в каждом составе равна 100 %")

    titanium_ok = all(
        item.composition.get("Ti", 0.0) >= config.GEN_MIN_TI for item in generated
    )
    check(titanium_ok, f"титан является основой (не менее {config.GEN_MIN_TI:g} %)")

    with session_scope() as session:
        existing = {
            tuple(sorted(alloy.composition.items()))
            for alloy in get_all_alloys(session)
        }
    novel_ok = all(
        tuple(sorted(item.composition.items())) not in existing for item in generated
    )
    check(novel_ok, "сгенерированные составы отсутствуют в базе")

    check(
        all(item.design_rationale and item.explanation for item in generated),
        "для каждого состава блока Б есть обоснование и оценка соответствия",
    )
    for item in generated:
        print(f"      {item.display_name()} — {item.percent:.1f} % — "
              f"Ti {item.composition.get('Ti', 0):.1f} %, "
              f"сумма {sum(item.composition.values()):.1f} %")

    # --- 5. Отчёт PDF --------------------------------------------------
    print("\n5. Экспорт отчёта")
    path = generate_pdf_report(
        query=query,
        similar=similar,
        generated=generated,
        summary="Отчёт сформирован в режиме самопроверки системы.",
        created_at=datetime.now(),
    )
    check(path.exists() and path.stat().st_size > 5000,
          f"PDF-отчёт создан: {path.name} ({path.stat().st_size} байт)")

    # --- Итог ----------------------------------------------------------
    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 70)
    if passed == total:
        print(f"РЕЗУЛЬТАТ: пройдено {passed} из {total} проверок. Система работоспособна.")
        print("=" * 70)
        return 0

    print(f"РЕЗУЛЬТАТ: пройдено {passed} из {total}. Не пройдены:")
    for ok, description in _results:
        if not ok:
            print(f"  - {description}")
    print("=" * 70)
    return 1
