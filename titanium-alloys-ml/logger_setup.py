# -*- coding: utf-8 -*-
"""
Настройка логирования для всего проекта.

Логи одновременно пишутся в файл logs/parser.log (с ротацией) и выводятся
в консоль. Функция get_logger() возвращает готовый логгер для модуля.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

import config

# Флаг, чтобы не настраивать корневой логгер повторно при каждом импорте
_configured = False


def _configure_root_logger() -> None:
    """Один раз настраивает корневой логгер: файл + консоль."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    # Обработчик записи в файл с ротацией по размеру
    file_handler = RotatingFileHandler(
        filename=str(config.PARSER_LOG),
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Обработчик вывода в консоль: показываем только предупреждения и ошибки,
    # чтобы не портить оформленный вывод rich в интерактивном режиме.
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Приглушаем слишком «болтливые» сторонние библиотеки
    for noisy in ("urllib3", "requests", "apscheduler", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Возвращает логгер с указанным именем, настроив систему при первом вызове."""
    _configure_root_logger()
    return logging.getLogger(name)


def enable_console_info() -> None:
    """
    Повышает подробность консольного вывода до INFO.
    Используется при ручном запуске парсинга, чтобы видеть прогресс.
    """
    _configure_root_logger()
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, RotatingFileHandler
        ):
            handler.setLevel(logging.INFO)
