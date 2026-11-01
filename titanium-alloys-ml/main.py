# -*- coding: utf-8 -*-
"""
Единая точка входа системы подбора и прогнозирования титановых сплавов.

Режимы запуска:
    python main.py              — консольный инструмент прогнозирования (модуль 2)
    python main.py --parse      — разовое наполнение/обновление базы (модуль 1)
    python main.py --schedule   — фоновое автообновление по расписанию
    python main.py --schedule --now — то же, но с немедленным первым запуском
    python main.py --stats      — статистика по базе данных
    python main.py --list       — вывод всех записей, сгруппированных по сплавам
    python main.py --selftest   — сквозная самопроверка системы
    python main.py --demo       — демонстрация без ввода с клавиатуры
"""

import argparse
import sys

# Гарантируем, что корень проекта есть в путях импорта,
# даже если main.py запущен из другого каталога.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from logger_setup import enable_console_info, get_logger  # noqa: E402

log = get_logger("main")


def cmd_parse() -> None:
    """Режим --parse: разовый запуск парсинга."""
    from parser.scraper import run_parsing

    enable_console_info()
    print("Запуск парсинга источников. Подробности пишутся в logs/parser.log\n")
    stats = run_parsing()
    print(
        f"\nГотово. Собрано записей: {stats['collected']}\n"
        f"  добавлено новых:            {stats['added']}\n"
        f"  новых вариантов состава:    {stats['variant']}\n"
        f"  пропущено как дубли:        {stats['duplicate']}\n"
        f"  источников с ошибкой:       {stats['failed_sources']}"
    )


def cmd_schedule(run_now: bool) -> None:
    """Режим --schedule: запуск планировщика автообновления."""
    from parser.scheduler import start_blocking

    enable_console_info()
    start_blocking(run_now=run_now)


def cmd_stats() -> None:
    """Режим --stats: краткая статистика по базе."""
    from database.models import get_stats, init_db, session_scope

    init_db()
    with session_scope() as session:
        stats = get_stats(session)

    last = stats["last_parsed"]
    print(f"Файл базы данных:   {config.DB_PATH}")
    print(f"Всего записей:      {stats['total']}")
    print(f"Уникальных сплавов: {stats['unique_names']}")
    print(f"Источников:         {stats['unique_sources']}")
    print(f"Последний парсинг:  "
          f"{last.strftime('%d.%m.%Y %H:%M') if last else '—'}")


def cmd_list() -> None:
    """Режим --list: вывод содержимого базы, сгруппированного по названию сплава."""
    from database.models import get_all_alloys, group_by_name, init_db, session_scope
    from ml.features import fmt, format_composition

    init_db()
    with session_scope() as session:
        groups = group_by_name(get_all_alloys(session))

    if not groups:
        print("База данных пуста. Выполните: python main.py --parse")
        return

    for name, records in groups:
        print(f"\n=== {name} ({len(records)} записей) " + "=" * 30)
        for alloy in records:
            print(f"  Источник: {alloy.source}")
            print(f"    Состав: {format_composition(alloy.composition)}")
            print(f"    Плотность: {fmt(alloy.density)} г/см³ | "
                  f"σв: {fmt(alloy.tensile_strength)} МПа | "
                  f"σ0,2: {fmt(alloy.yield_strength)} МПа | "
                  f"E: {fmt(alloy.elastic_modulus)} ГПа")
            print(f"    HB: {fmt(alloy.hardness_brinell)} | "
                  f"Tпл: {fmt(alloy.melting_point)} °C | "
                  f"λ: {fmt(alloy.thermal_conductivity)} Вт/(м·К) | "
                  f"σ-1: {fmt(alloy.fatigue_strength)} МПа")
            print(f"    Коррозионная стойкость: {alloy.corrosion_resistance or '—'}")
            print(f"    Применение: {alloy.application or '—'}")
            print(f"    Дата парсинга: {alloy.parsed_at:%d.%m.%Y %H:%M}")


def cmd_demo() -> None:
    """
    Режим --demo: показ возможностей системы без ввода с клавиатуры.

    Используется при запуске «docker compose up»: Compose подключает
    к контейнеру только вывод, поэтому интерактивное меню там недоступно.
    """
    from database.models import init_db
    from interface.console import run_demo

    init_db()
    run_demo()


def cmd_console() -> None:
    """Режим по умолчанию: интерактивный инструмент прогнозирования."""
    from database.models import get_stats, init_db, session_scope
    from interface.console import main as console_main

    init_db()
    with session_scope() as session:
        stats = get_stats(session)

    # Если база пуста — предлагаем наполнить её сразу, не выходя из программы
    if stats["total"] == 0:
        print("База данных пуста.")
        answer = input("Выполнить наполнение базы сейчас? (д/н): ").strip().lower()
        if answer in ("д", "да", "y", "yes"):
            cmd_parse()
            print()
        else:
            print("Запустите наполнение командой: python main.py --parse")
            return

    console_main()


def build_arg_parser() -> argparse.ArgumentParser:
    """Описание аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Система подбора и прогнозирования титановых сплавов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--parse", action="store_true",
                        help="разовый запуск парсинга источников")
    parser.add_argument("--schedule", action="store_true",
                        help="запуск планировщика автообновления базы")
    parser.add_argument("--now", action="store_true",
                        help="вместе с --schedule: сразу выполнить первый парсинг")
    parser.add_argument("--stats", action="store_true",
                        help="показать статистику базы данных")
    parser.add_argument("--list", action="store_true",
                        help="вывести все записи, сгруппированные по сплавам")
    parser.add_argument("--selftest", action="store_true",
                        help="сквозная самопроверка системы")
    parser.add_argument("--demo", action="store_true",
                        help="автоматическая демонстрация без ввода с клавиатуры")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        if args.parse:
            cmd_parse()
        elif args.schedule:
            cmd_schedule(run_now=args.now)
        elif args.stats:
            cmd_stats()
        elif args.list:
            cmd_list()
        elif args.selftest:
            import selftest

            enable_console_info()
            sys.exit(selftest.run())
        elif args.demo:
            cmd_demo()
        else:
            cmd_console()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    except Exception as exc:
        log.exception("Необработанная ошибка: %s", exc)
        print(f"\nОшибка: {exc}\nПодробности в файле {config.PARSER_LOG}")
        sys.exit(1)


if __name__ == "__main__":
    main()
