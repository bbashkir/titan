# -*- coding: utf-8 -*-
"""
Консольный интерфейс прогнозирования титановых сплавов.

Пользователь выбирает нужные свойства цифрами и задаёт диапазоны;
незаданные параметры при поиске игнорируются. После расчёта выводятся
два блока — существующие сплавы и сгенерированные моделью — и
предлагается сохранить отчёт в PDF.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import config
from database.models import get_stats, init_db, session_scope
from logger_setup import get_logger
from ml.features import (
    CORROSION_LABELS_RU,
    UserQuery,
    build_dataframe,
    fmt,
    format_composition,
)
from ml.generative_model import GenerativeModel
from ml.similarity_model import SimilarityModel
from reports.pdf_generator import generate_pdf_report

log = get_logger("interface.console")
console = Console()

# Пункты меню: номер -> (поле БД, подпись, единица измерения)
MENU_ITEMS = {
    "1": ("density", "Плотность", "г/см³"),
    "2": ("tensile_strength", "Прочность на растяжение", "МПа"),
    "3": ("elastic_modulus", "Модуль упругости", "ГПа"),
    "4": ("hardness_brinell", "Твёрдость по Бринеллю", "HB"),
    "5": ("melting_point", "Температура плавления", "°C"),
    "6": ("corrosion_resistance", "Коррозионная стойкость", "low / medium / high"),
    "7": ("fatigue_strength", "Усталостная прочность", "МПа"),
    "8": ("application", "Область применения", "текст"),
}

# Символы завершения ввода критериев
DONE_KEYS = {"0", "9"}


# ==========================================================================
# ВСПОМОГАТЕЛЬНЫЙ ВВОД
# ==========================================================================

def ask_float(prompt: str) -> Optional[float]:
    """
    Запрашивает число. Пустая строка означает «ограничение не задано».
    При некорректном вводе повторяет вопрос.
    """
    while True:
        raw = console.input(f"  {prompt}").strip().replace(",", ".")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            console.print("  [red]Введите число или оставьте поле пустым.[/red]")


def ask_range(label: str, unit: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """
    Запрашивает диапазон «мин / макс».
    Возвращает None, если пользователь не задал ни одной границы.
    """
    console.print(f"\n[bold cyan]{label}[/bold cyan] ({unit}). "
                  f"Пустое поле — ограничение не задавать.")
    minimum = ask_float(f"минимум ({unit}): ")
    maximum = ask_float(f"максимум ({unit}): ")

    if minimum is None and maximum is None:
        console.print("  [yellow]Критерий не задан — параметр будет проигнорирован.[/yellow]")
        return None
    if minimum is not None and maximum is not None and minimum > maximum:
        console.print("  [yellow]Минимум больше максимума — значения переставлены.[/yellow]")
        minimum, maximum = maximum, minimum
    return minimum, maximum


def ask_corrosion() -> Optional[str]:
    """Запрашивает требуемый уровень коррозионной стойкости."""
    console.print("\n[bold cyan]Коррозионная стойкость[/bold cyan]")
    console.print("  1 — low (низкая)\n  2 — medium (средняя)\n  3 — high (высокая)")
    mapping = {"1": "low", "2": "medium", "3": "high",
               "low": "low", "medium": "medium", "high": "high"}
    while True:
        raw = console.input("  Ваш выбор (Enter — не задавать): ").strip().lower()
        if not raw:
            return None
        if raw in mapping:
            return mapping[raw]
        console.print("  [red]Введите 1, 2, 3 или название уровня.[/red]")


def ask_application() -> Optional[str]:
    """Запрашивает область применения текстом."""
    console.print("\n[bold cyan]Область применения[/bold cyan] "
                  "(например: авиация, медицина, судостроение)")
    raw = console.input("  Введите текст (Enter — не задавать): ").strip()
    return raw or None


# ==========================================================================
# ВЫВОД
# ==========================================================================

def print_banner() -> None:
    """Шапка программы."""
    console.print()
    console.print(Panel.fit(
        Text.from_markup(
            "[bold white]СИСТЕМА ПОДБОРА И ПРОГНОЗИРОВАНИЯ ТИТАНОВЫХ СПЛАВОВ[/bold white]\n"
            "[dim]Модуль 2 — консольный инструмент прогнозирования[/dim]"
        ),
        border_style="cyan",
        padding=(1, 4),
    ))


def print_db_stats() -> None:
    """Краткая сводка по содержимому базы данных."""
    with session_scope() as session:
        stats = get_stats(session)
    last = stats["last_parsed"]
    last_text = last.strftime("%d.%m.%Y %H:%M") if last else "—"
    console.print(
        f"[dim]База данных: записей — {stats['total']}, "
        f"уникальных сплавов — {stats['unique_names']}, "
        f"источников — {stats['unique_sources']}, "
        f"последнее обновление — {last_text}[/dim]\n"
    )


def print_menu(query: UserQuery) -> None:
    """Меню выбора критериев с отметками уже заданных параметров."""
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("№", width=3)
    table.add_column("Параметр", width=32)
    table.add_column("Единицы", width=18)
    table.add_column("Текущее значение", style="green")

    for key, (field_name, label, unit) in MENU_ITEMS.items():
        if field_name == "corrosion_resistance":
            current = (
                f"не ниже «{CORROSION_LABELS_RU[query.corrosion]}»"
                if query.corrosion else "—"
            )
        elif field_name == "application":
            current = query.application or "—"
        else:
            bounds = query.ranges.get(field_name)
            if bounds:
                vmin, vmax = bounds
                if vmin is not None and vmax is not None:
                    current = f"{fmt(vmin)} … {fmt(vmax)}"
                elif vmin is not None:
                    current = f"от {fmt(vmin)}"
                else:
                    current = f"до {fmt(vmax)}"
            else:
                current = "—"
        table.add_row(key, label, unit, current)

    console.print(Rule("[bold]Выбор требуемых свойств[/bold]", style="cyan"))
    console.print(table)
    console.print(
        "\n  [bold]0[/bold] — Готово, запустить прогнозирование     "
        "[bold]с[/bold] — сбросить все критерии     "
        "[bold]в[/bold] — выход\n"
    )


def print_similar_block(results: List) -> None:
    """БЛОК А — существующие сплавы из базы."""
    console.print()
    console.print(Rule(
        "[bold white on blue] БЛОК А — ПОХОЖИЕ СУЩЕСТВУЮЩИЕ СПЛАВЫ [/bold white on blue]",
        style="blue",
    ))

    if not results:
        console.print("[yellow]В базе не нашлось подходящих записей.[/yellow]")
        return

    for position, item in enumerate(results, start=1):
        header = (
            f"[bold]{position}. {item.name}[/bold]   "
            f"[green]соответствие {item.percent:.1f} %[/green]"
        )
        console.print(f"\n{header}")
        console.print(f"   [dim]Источник:[/dim] {item.source}")
        console.print(f"   [dim]Состав:[/dim]   {format_composition(item.composition)}")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Свойство", style="cyan", width=28)
        table.add_column("Значение", width=18)

        for prop in config.NUMERIC_PROPERTIES:
            label, unit = config.PROPERTY_LABELS[prop]
            value = item.properties.get(prop)
            table.add_row(label, f"{fmt(value)} {unit}".strip() if value is not None else "—")

        corrosion = item.corrosion_resistance
        table.add_row(
            "Коррозионная стойкость",
            CORROSION_LABELS_RU.get(corrosion, corrosion) if corrosion else "—",
        )
        table.add_row("Область применения", item.application or "—")
        console.print(table)
        console.print(f"   [italic]{item.explanation}[/italic]")


def print_generated_block(results: List) -> None:
    """БЛОК Б — сплавы, сгенерированные моделью."""
    console.print()
    console.print(Rule(
        "[bold white on magenta] БЛОК Б — НОВЫЕ СГЕНЕРИРОВАННЫЕ СПЛАВЫ [/bold white on magenta]",
        style="magenta",
    ))

    if not results:
        console.print("[yellow]Модель не смогла предложить корректные новые составы.[/yellow]")
        return

    for position, item in enumerate(results, start=1):
        console.print(
            f"\n[bold]{position}. {item.display_name()}[/bold]   "
            f"[green]соответствие {item.percent:.1f} %[/green]"
        )
        console.print(f"   [dim]Состав:[/dim] {format_composition(item.composition)}")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Свойство (прогноз)", style="magenta", width=28)
        table.add_column("Значение", width=18)
        for prop in config.NUMERIC_PROPERTIES:
            label, unit = config.PROPERTY_LABELS[prop]
            value = item.properties.get(prop)
            table.add_row(label, f"{fmt(value)} {unit}".strip() if value is not None else "—")
        corrosion = item.corrosion_resistance
        table.add_row(
            "Коррозионная стойкость",
            CORROSION_LABELS_RU.get(corrosion, corrosion) if corrosion else "—",
        )
        table.add_row("Область применения", item.application or "—")
        console.print(table)

        console.print(f"   [italic]{item.explanation}[/italic]")
        console.print(f"   [dim]Обоснование состава:[/dim] {item.design_rationale}")
        if item.advantage:
            console.print(f"   [dim]Сравнение с аналогом:[/dim] {item.advantage}")
        console.print(f"   [bold yellow]{item.note}[/bold yellow]")


def print_summary(similar: List, generated: List) -> str:
    """
    Выводит и возвращает краткое резюме: какой сплав оптимален и почему.
    Тот же текст попадает в PDF-отчёт.
    """
    if not similar and not generated:
        return "Подходящих вариантов не найдено."

    best_existing = similar[0] if similar else None
    best_generated = generated[0] if generated else None

    lines: List[str] = []
    if best_existing and (not best_generated or
                          best_existing.percent >= best_generated.percent):
        lines.append(
            f"Оптимальный выбор — существующий сплав {best_existing.name} "
            f"(соответствие {best_existing.percent:.1f} %, источник: {best_existing.source})."
        )
        lines.append(
            "Он доступен как промышленная марка с подтверждёнными "
            "характеристиками, поэтому не требует опытной выплавки и аттестации."
        )
        if best_generated:
            lines.append(
                f"Сгенерированный состав {best_generated.display_name()} "
                f"(соответствие {best_generated.percent:.1f} %) может рассматриваться "
                f"как направление дальнейшей разработки."
            )
    else:
        lines.append(
            f"Наилучшее расчётное соответствие даёт сгенерированный состав "
            f"{best_generated.display_name()} — {best_generated.percent:.1f} %."
        )
        lines.append(best_generated.advantage or "")
        lines.append(
            "Состав в базе отсутствует: значения свойств являются прогнозом модели "
            "и требуют экспериментальной проверки."
        )
        if best_existing:
            lines.append(
                f"Среди промышленных марок ближе всего {best_existing.name} "
                f"({best_existing.percent:.1f} %) — его следует принять как базовый вариант."
            )

    summary = " ".join(line for line in lines if line)
    console.print()
    console.print(Panel(summary, title="[bold]Краткое резюме[/bold]",
                        border_style="green", padding=(1, 2)))
    return summary


# ==========================================================================
# ОСНОВНОЙ СЦЕНАРИЙ
# ==========================================================================

class ConsoleApp:
    """Приложение: обучает модели один раз и обслуживает запросы пользователя."""

    def __init__(self):
        self.similarity: Optional[SimilarityModel] = None
        self.generative: Optional[GenerativeModel] = None

    def prepare_models(self) -> bool:
        """
        Загружает данные и обучает обе модели.
        Возвращает False, если база пуста и работать не с чем.
        """
        init_db()

        with session_scope() as session:
            df = build_dataframe(session)

        if df.empty:
            console.print(
                "[bold red]База данных пуста.[/bold red] "
                "Сначала выполните наполнение базы командой:\n"
                "    [cyan]python main.py --parse[/cyan]\n"
            )
            return False

        with console.status("[cyan]Обучение модели поиска похожих сплавов…[/cyan]"):
            self.similarity = SimilarityModel().fit(df)

        with console.status("[magenta]Обучение генеративной модели (VAE)…[/magenta]"):
            # Регрессоры свойств переиспользуем из первой модели — они уже обучены
            self.generative = GenerativeModel(predictor=self.similarity.predictor).fit(df)
            self.generative.save()

        console.print("[green]Модели обучены.[/green]\n")
        return True

    def collect_query(self) -> Optional[UserQuery]:
        """
        Диалог выбора критериев.
        Возвращает готовый запрос или None, если пользователь вышел.
        """
        query = UserQuery()

        while True:
            print_menu(query)
            choice = console.input("[bold]Выберите пункт:[/bold] ").strip().lower()

            if choice in DONE_KEYS:
                if query.is_empty():
                    console.print("[yellow]Не задан ни один критерий. "
                                  "Выберите хотя бы один параметр.[/yellow]")
                    continue
                return query

            if choice in ("в", "q", "exit", "quit"):
                return None

            if choice in ("с", "c", "reset"):
                query = UserQuery()
                console.print("[green]Критерии сброшены.[/green]")
                continue

            if choice not in MENU_ITEMS:
                console.print("[red]Неизвестный пункт меню.[/red]")
                continue

            field_name, label, unit = MENU_ITEMS[choice]

            if field_name == "corrosion_resistance":
                query.corrosion = ask_corrosion()
            elif field_name == "application":
                query.application = ask_application()
            else:
                bounds = ask_range(label, unit)
                if bounds is None:
                    query.ranges.pop(field_name, None)
                else:
                    query.ranges[field_name] = bounds

    def run_prediction(self, query: UserQuery, interactive: bool = True) -> None:
        """
        Выполняет прогнозирование и выводит результаты.

        interactive=False — режим демонстрации: отчёт сохраняется в PDF
        автоматически, без вопроса пользователю.
        """
        console.print()
        console.print(Panel(
            "\n".join(f"• {line}" for line in query.describe()),
            title="[bold]Параметры запроса[/bold]",
            border_style="cyan",
            padding=(1, 2),
        ))

        with console.status("[cyan]Поиск похожих сплавов в базе…[/cyan]"):
            similar = self.similarity.find_similar(query, top_n=config.TOP_SIMILAR)

        with console.status("[magenta]Генерация новых составов…[/magenta]"):
            generated = self.generative.generate(
                query, top_n=config.TOP_GENERATED, similarity_model=self.similarity
            )

        print_similar_block(similar)
        print_generated_block(generated)
        summary = print_summary(similar, generated)

        if interactive:
            self.offer_pdf(query, similar, generated, summary)
        else:
            self.save_pdf(query, similar, generated, summary)

    @staticmethod
    def save_pdf(query: UserQuery, similar: List, generated: List,
                 summary: str) -> None:
        """Сохраняет отчёт без вопроса — используется в режиме демонстрации."""
        try:
            path = generate_pdf_report(
                query=query, similar=similar, generated=generated,
                summary=summary, created_at=datetime.now(),
            )
        except Exception as exc:
            log.exception("Ошибка формирования PDF: %s", exc)
            console.print(f"[red]Не удалось сохранить отчёт: {exc}[/red]")
            return
        console.print(f"\n[green]Отчёт сохранён:[/green] {path}")

    @staticmethod
    def offer_pdf(query: UserQuery, similar: List, generated: List,
                  summary: str) -> None:
        """Предлагает сохранить отчёт в PDF."""
        console.print()
        answer = console.input(
            "[bold]Сохранить отчёт в PDF? (д/н):[/bold] "
        ).strip().lower()

        if answer not in ("д", "да", "y", "yes"):
            console.print("[dim]Отчёт не сохранён.[/dim]")
            return

        try:
            path = generate_pdf_report(
                query=query,
                similar=similar,
                generated=generated,
                summary=summary,
                created_at=datetime.now(),
            )
        except Exception as exc:
            log.exception("Ошибка формирования PDF: %s", exc)
            console.print(f"[red]Не удалось сохранить отчёт: {exc}[/red]")
            return

        console.print(f"[green]Отчёт сохранён:[/green] {path}")

    def run(self) -> None:
        """Главный цикл приложения."""
        print_banner()
        print_db_stats()

        if not self.prepare_models():
            return

        while True:
            query = self.collect_query()
            if query is None:
                console.print("\n[dim]Работа завершена.[/dim]")
                return

            try:
                self.run_prediction(query)
            except Exception as exc:
                log.exception("Ошибка прогнозирования: %s", exc)
                console.print(f"[red]Ошибка при расчёте: {exc}[/red]")

            console.print()
            again = console.input(
                "[bold]Выполнить ещё один подбор? (д/н):[/bold] "
            ).strip().lower()
            if again not in ("д", "да", "y", "yes"):
                console.print("\n[dim]Работа завершена.[/dim]")
                return


# Заранее заготовленные запросы для режима демонстрации.
# Подобраны так, чтобы показать разные сценарии работы системы:
# высокопрочный конструкционный сплав и биосовместимый сплав
# с низким модулем упругости.
DEMO_QUERIES = [
    (
        "Сценарий 1. Силовая деталь авиационной конструкции",
        UserQuery(
            ranges={
                "tensile_strength": (900.0, None),
                "density": (None, 4.6),
            },
            corrosion="high",
            application="авиация",
        ),
    ),
    (
        "Сценарий 2. Имплантат: низкий модуль упругости при высокой прочности",
        UserQuery(
            ranges={
                "elastic_modulus": (None, 90.0),
                "tensile_strength": (800.0, None),
            },
            corrosion="high",
            application="медицина",
        ),
    ),
]


def run_demo() -> None:
    """
    Автоматическая демонстрация без ввода с клавиатуры.

    Нужна для запуска в контейнере командой «docker compose up»: Compose
    подключает к контейнеру только вывод, но не ввод, поэтому интерактивное
    меню там работать не может. Демонстрация проходит те же этапы —
    обучение моделей, оба блока результата, объяснения, экспорт PDF —
    на заранее заданных запросах.
    """
    print_banner()
    print_db_stats()

    application = ConsoleApp()
    if not application.prepare_models():
        return

    for title, query in DEMO_QUERIES:
        console.print()
        console.print(Rule(f"[bold yellow]{title}[/bold yellow]", style="yellow"))
        application.run_prediction(query, interactive=False)

    console.print()
    console.print(Panel(
        Text.from_markup(
            "Демонстрация завершена. Отчёты сохранены в папку "
            "[cyan]reports/output[/cyan].\n\n"
            "Интерактивный подбор с выбором параметров вручную:\n"
            "  [bold cyan]docker compose run --rm app[/bold cyan]\n\n"
            "Самопроверка системы:\n"
            "  [bold cyan]docker compose run --rm app --selftest[/bold cyan]"
        ),
        border_style="green",
        padding=(1, 2),
    ))


def main() -> None:
    """Точка входа консольного интерфейса."""
    try:
        ConsoleApp().run()
    except KeyboardInterrupt:
        console.print("\n[dim]Прервано пользователем.[/dim]")


if __name__ == "__main__":
    main()
