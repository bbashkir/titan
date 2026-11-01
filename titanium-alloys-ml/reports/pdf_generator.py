# -*- coding: utf-8 -*-
"""
Формирование PDF-отчёта по результатам прогнозирования.

Отчёт содержит:
  * дату формирования и параметры запроса пользователя;
  * БЛОК А — похожие существующие сплавы с полными данными и объяснениями;
  * БЛОК Б — сгенерированные сплавы с прогнозом свойств и обоснованием;
  * краткое резюме — какой сплав оптимален и почему.

Используется reportlab. Для корректного вывода кириллицы регистрируется
системный TTF-шрифт (список путей — в config.PDF_FONT_CANDIDATES).
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config
from logger_setup import get_logger
from ml.features import CORROSION_LABELS_RU, UserQuery, fmt, format_composition

log = get_logger("reports.pdf_generator")

# Имена зарегистрированных шрифтов (заполняются в _register_fonts)
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
_fonts_registered = False


def esc(text) -> str:
    """
    Экранирует спецсимволы для разметки reportlab.

    Абзацы Paragraph разбираются как мини-XML, поэтому символы «&», «<», «>»
    в данных (например, в URL источника с параметрами) нужно экранировать,
    иначе построение документа завершится ошибкой.
    """
    if text is None:
        return "—"
    return xml_escape(str(text))


def _register_fonts() -> None:
    """
    Регистрирует TTF-шрифт с поддержкой кириллицы.

    Проверяет пути из config.PDF_FONT_CANDIDATES. Если ни один шрифт
    не найден, остаётся встроенный Helvetica — латиница выведется
    корректно, кириллица нет, о чём пишется предупреждение в лог.
    """
    global FONT_REGULAR, FONT_BOLD, _fonts_registered
    if _fonts_registered:
        return

    regular_path = next(
        (path for path in config.PDF_FONT_CANDIDATES if Path(path).exists()), None
    )
    if regular_path is None:
        log.warning(
            "Не найден TTF-шрифт с кириллицей. Проверьте config.PDF_FONT_CANDIDATES. "
            "Отчёт будет сформирован встроенным шрифтом без поддержки кириллицы."
        )
        _fonts_registered = True
        return

    pdfmetrics.registerFont(TTFont("ReportFont", regular_path))
    FONT_REGULAR = "ReportFont"

    bold_path = next(
        (path for path in config.PDF_FONT_BOLD_CANDIDATES if Path(path).exists()), None
    )
    if bold_path:
        pdfmetrics.registerFont(TTFont("ReportFont-Bold", bold_path))
        FONT_BOLD = "ReportFont-Bold"
    else:
        # Жирного начертания нет — используем обычное
        FONT_BOLD = FONT_REGULAR

    log.info("Шрифт отчёта: %s", regular_path)
    _fonts_registered = True


def _build_styles():
    """Готовит набор стилей абзацев для отчёта."""
    _register_fonts()
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "TitleRu", parent=base["Title"], fontName=FONT_BOLD,
            fontSize=17, leading=22, alignment=TA_CENTER, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleRu", parent=base["Normal"], fontName=FONT_REGULAR,
            fontSize=9.5, leading=13, alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"), spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1Ru", parent=base["Heading1"], fontName=FONT_BOLD,
            fontSize=13, leading=17, spaceBefore=14, spaceAfter=8,
            textColor=colors.HexColor("#0B3D91"),
        ),
        "h2": ParagraphStyle(
            "H2Ru", parent=base["Heading2"], fontName=FONT_BOLD,
            fontSize=11, leading=15, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyRu", parent=base["Normal"], fontName=FONT_REGULAR,
            fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "SmallRu", parent=base["Normal"], fontName=FONT_REGULAR,
            fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#333333"), spaceAfter=2,
        ),
        "note": ParagraphStyle(
            "NoteRu", parent=base["Normal"], fontName=FONT_BOLD,
            fontSize=9, leading=12, textColor=colors.HexColor("#B35C00"),
            spaceBefore=3, spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "CellRu", parent=base["Normal"], fontName=FONT_REGULAR,
            fontSize=8.5, leading=11,
        ),
    }
    return styles


def _properties_table(properties: dict, corrosion: Optional[str],
                      application: Optional[str], styles,
                      predicted: bool = False) -> Table:
    """
    Таблица свойств сплава. Отсутствующие значения выводятся прочерком.
    predicted=True — свойства получены расчётом, заголовок это отражает.
    """
    header_value = "Прогноз модели" if predicted else "Значение"
    rows = [["Свойство", header_value]]

    for prop in config.NUMERIC_PROPERTIES:
        label, unit = config.PROPERTY_LABELS[prop]
        value = properties.get(prop)
        text = f"{fmt(value)} {unit}".strip() if value is not None else "—"
        rows.append([label, text])

    rows.append([
        "Коррозионная стойкость",
        CORROSION_LABELS_RU.get(corrosion, corrosion) if corrosion else "—",
    ])
    rows.append([
        "Область применения",
        Paragraph(esc(application) if application else "—", styles["cell"]),
    ])

    accent = colors.HexColor("#7B2D8E") if predicted else colors.HexColor("#0B3D91")
    table = Table(rows, colWidths=[62 * mm, 108 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F4F8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8BFC9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _add_page_number(canvas, doc) -> None:
    """Колонтитул с номером страницы."""
    canvas.saveState()
    canvas.setFont(FONT_REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"— {doc.page} —")
    canvas.restoreState()


def generate_pdf_report(query: UserQuery,
                        similar: List,
                        generated: List,
                        summary: str,
                        created_at: Optional[datetime] = None,
                        output_path: Optional[Path] = None) -> Path:
    """
    Формирует PDF-отчёт и возвращает путь к созданному файлу.

    query     — параметры запроса пользователя;
    similar   — список SimilarAlloy (БЛОК А);
    generated — список GeneratedAlloy (БЛОК Б);
    summary   — текст краткого резюме.
    """
    created_at = created_at or datetime.now()
    styles = _build_styles()

    if output_path is None:
        filename = config.PDF_FILENAME_TEMPLATE.format(
            timestamp=created_at.strftime("%Y%m%d_%H%M%S")
        )
        output_path = config.REPORTS_DIR / filename
    output_path = Path(output_path)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="Отчёт по подбору титановых сплавов",
        author="Система подбора и прогнозирования титановых сплавов",
    )

    story = []

    # ------------------------------------------------------------ титул
    story.append(Paragraph("Отчёт по подбору и прогнозированию<br/>титановых сплавов",
                           styles["title"]))
    story.append(Paragraph(
        f"Сформирован {created_at.strftime('%d.%m.%Y в %H:%M')}",
        styles["subtitle"],
    ))

    # -------------------------------------------------- параметры запроса
    story.append(Paragraph("Параметры запроса пользователя", styles["h1"]))
    criteria = query.describe()
    if criteria:
        rows = [[Paragraph(f"{i}.", styles["cell"]), Paragraph(esc(line), styles["cell"])]
                for i, line in enumerate(criteria, start=1)]
        table = Table(rows, colWidths=[10 * mm, 160 * mm], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("Критерии не заданы.", styles["body"]))

    # ---------------------------------------------------------- БЛОК А
    story.append(Paragraph(
        "БЛОК А. Похожие существующие сплавы (по данным базы)", styles["h1"]
    ))

    if not similar:
        story.append(Paragraph("Подходящих записей в базе данных не найдено.",
                               styles["body"]))
    else:
        for position, item in enumerate(similar, start=1):
            block = [
                Paragraph(
                    f"{position}. {esc(item.name)} — соответствие запросу "
                    f"{item.percent:.1f} %",
                    styles["h2"],
                ),
                Paragraph(f"<b>Источник:</b> {esc(item.source)}", styles["small"]),
                Paragraph(
                    "<b>Химический состав:</b> "
                    f"{esc(format_composition(item.composition))}",
                    styles["small"],
                ),
                Spacer(1, 3),
                _properties_table(
                    item.properties, item.corrosion_resistance,
                    item.application, styles, predicted=False,
                ),
                Spacer(1, 4),
                Paragraph(f"<b>Объяснение:</b> {esc(item.explanation)}", styles["small"]),
                Spacer(1, 10),
            ]
            story.append(KeepTogether(block))

    # ---------------------------------------------------------- БЛОК Б
    story.append(PageBreak())
    story.append(Paragraph(
        "БЛОК Б. Новые сплавы, сгенерированные моделью", styles["h1"]
    ))
    story.append(Paragraph(
        "Составы получены вариационным автокодировщиком, обученным на данных базы. "
        "Сумма долей элементов приведена к 100 %, содержание титана как основы "
        f"не ниже {fmt(config.GEN_MIN_TI)} %. Значения свойств рассчитаны "
        "регрессионными моделями и требуют экспериментальной проверки.",
        styles["small"],
    ))
    story.append(Spacer(1, 6))

    if not generated:
        story.append(Paragraph(
            "Модель не смогла предложить корректные новые составы.", styles["body"]
        ))
    else:
        for position, item in enumerate(generated, start=1):
            block = [
                Paragraph(
                    f"{position}. {esc(item.display_name())} — соответствие запросу "
                    f"{item.percent:.1f} %",
                    styles["h2"],
                ),
                Paragraph(
                    "<b>Химический состав:</b> "
                    f"{esc(format_composition(item.composition))}",
                    styles["small"],
                ),
                Spacer(1, 3),
                _properties_table(
                    item.properties, item.corrosion_resistance,
                    item.application, styles, predicted=True,
                ),
                Spacer(1, 4),
                Paragraph(f"<b>Соответствие запросу:</b> {esc(item.explanation)}",
                          styles["small"]),
                Paragraph(f"<b>Обоснование состава:</b> {esc(item.design_rationale)}",
                          styles["small"]),
            ]
            if item.advantage:
                block.append(
                    Paragraph(f"<b>Сравнение с аналогом:</b> {esc(item.advantage)}",
                              styles["small"])
                )
            # Без пиктограмм: в системных шрифтах Windows нет знака «⚠»,
            # и вместо него в PDF печатается пустой прямоугольник.
            block.append(Paragraph(esc(item.note), styles["note"]))
            block.append(Spacer(1, 8))
            story.append(KeepTogether(block))

    # --------------------------------------------------------- резюме
    story.append(Paragraph("Краткое резюме", styles["h1"]))
    summary_table = Table([[Paragraph(esc(summary), styles["body"])]],
                          colWidths=[174 * mm], hAlign="LEFT")
    summary_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1B7F3B")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F8F2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Отчёт сформирован автоматически. Данные блока А получены из открытых "
        "источников и нормативной документации; данные блока Б являются "
        "результатом машинного прогноза и не заменяют испытаний материала.",
        styles["small"],
    ))

    document.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    log.info("PDF-отчёт сохранён: %s", output_path)
    return output_path
