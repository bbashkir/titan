# -*- coding: utf-8 -*-
"""
Планировщик автоматического обновления базы данных.

По умолчанию парсинг запускается раз в месяц (1-го числа в 3:00).
Интервал настраивается в config.py: SCHEDULE_INTERVAL, SCHEDULE_DAY_OF_MONTH,
SCHEDULE_HOUR, SCHEDULE_MINUTE.

Правила обновления:
  * новые записи добавляются;
  * существующие записи не изменяются и не удаляются;
  * в лог пишется, сколько записей добавлено и сколько пропущено как дубли.

Запуск: python main.py --schedule
Остановка: Ctrl+C.
"""

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from logger_setup import get_logger
from parser.scraper import run_parsing

log = get_logger("parser.scheduler")


def scheduled_job() -> None:
    """
    Задание, выполняемое по расписанию.
    Оборачивает run_parsing() так, чтобы любая ошибка была записана в лог,
    но не «уронила» планировщик.
    """
    log.info("### Плановое обновление базы данных запущено (%s)",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        stats = run_parsing()
        log.info(
            "### Плановое обновление завершено. Добавлено новых записей: %d, "
            "новых вариантов состава: %d, пропущено дублей: %d",
            stats["added"], stats["variant"], stats["duplicate"],
        )
    except Exception as exc:
        log.exception("### Плановое обновление завершилось ошибкой: %s", exc)


def build_trigger() -> CronTrigger:
    """Формирует триггер расписания на основе настроек из config.py."""
    interval = config.SCHEDULE_INTERVAL.lower()

    if interval == "monthly":
        return CronTrigger(
            day=config.SCHEDULE_DAY_OF_MONTH,
            hour=config.SCHEDULE_HOUR,
            minute=config.SCHEDULE_MINUTE,
        )
    if interval == "weekly":
        return CronTrigger(
            day_of_week="mon",
            hour=config.SCHEDULE_HOUR,
            minute=config.SCHEDULE_MINUTE,
        )
    if interval == "daily":
        return CronTrigger(hour=config.SCHEDULE_HOUR, minute=config.SCHEDULE_MINUTE)

    log.warning("Неизвестный интервал %r, используется ежемесячный",
                config.SCHEDULE_INTERVAL)
    return CronTrigger(
        day=config.SCHEDULE_DAY_OF_MONTH,
        hour=config.SCHEDULE_HOUR,
        minute=config.SCHEDULE_MINUTE,
    )


def start_blocking(run_now: bool = False) -> None:
    """
    Запускает планировщик в блокирующем режиме — процесс работает,
    пока пользователь не нажмёт Ctrl+C.

    run_now=True — дополнительно выполнить парсинг сразу при старте.
    """
    trigger = build_trigger()
    scheduler = BlockingScheduler(timezone=None)
    scheduler.add_job(
        scheduled_job,
        trigger=trigger,
        id="titanium_parsing",
        name="Обновление базы титановых сплавов",
        replace_existing=True,
        misfire_grace_time=3600,  # если ПК был выключен — запустить в течение часа
        coalesce=True,            # пропущенные запуски объединять в один
    )

    next_run = trigger.get_next_fire_time(None, datetime.now())
    log.info("Планировщик запущен. Режим: %s. Следующий запуск: %s",
             config.SCHEDULE_INTERVAL, next_run)
    print(f"Планировщик запущен (режим: {config.SCHEDULE_INTERVAL}).")
    print(f"Следующее обновление базы: {next_run}")
    print("Для остановки нажмите Ctrl+C.\n")

    if run_now:
        scheduled_job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Планировщик остановлен пользователем")
        print("\nПланировщик остановлен.")


def start_background() -> BackgroundScheduler:
    """
    Запускает планировщик в фоновом потоке и возвращает его объект.
    Используется, если консольный интерфейс работает одновременно
    с автообновлением базы.
    """
    scheduler = BackgroundScheduler(timezone=None)
    scheduler.add_job(
        scheduled_job,
        trigger=build_trigger(),
        id="titanium_parsing",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.start()
    log.info("Фоновый планировщик запущен")
    return scheduler


if __name__ == "__main__":
    # Ручной запуск планировщика: python -m parser.scheduler
    from logger_setup import enable_console_info

    enable_console_info()
    start_blocking()
