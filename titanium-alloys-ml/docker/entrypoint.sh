#!/bin/sh
# Точка входа контейнера.
#
# Задача: при самом первом запуске база данных ещё не наполнена, и приложение
# упёрлось бы в вопрос «база пуста». Поэтому здесь проверяется наличие записей
# и при необходимости выполняется парсинг — чтобы «docker compose up» сразу
# приводил к рабочей системе.
#
# Все аргументы команды передаются в main.py без изменений:
#   docker compose run --rm app --stats   ->   python main.py --stats

set -e

cd /app

# Проверяем, есть ли в базе записи. Отдельный маленький скрипт вместо
# проверки размера файла: файл может существовать, но быть пустым.
RECORDS=$(python - <<'PY'
import sys
try:
    from database.models import get_stats, init_db, session_scope
    init_db()
    with session_scope() as session:
        print(get_stats(session)["total"])
except Exception:
    # Любая ошибка на этом этапе — считаем, что записей нет,
    # и пробуем наполнить базу обычным путём.
    print(0)
PY
)

if [ "$RECORDS" -eq 0 ] 2>/dev/null; then
    echo "-------------------------------------------------------------"
    echo " Первый запуск: база данных пуста, выполняется наполнение."
    echo " Подробности пишутся в logs/parser.log"
    echo "-------------------------------------------------------------"
    python main.py --parse
    echo
else
    echo "База данных готова: записей — $RECORDS"
    echo
fi

exec python main.py "$@"
