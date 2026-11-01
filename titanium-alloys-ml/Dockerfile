# Образ системы подбора и прогнозирования титановых сплавов.
#
# Сборка:  docker compose build
# Запуск:  docker compose up
#
# Приложение консольное и интерактивное, поэтому контейнеру нужен терминал
# (stdin_open + tty в docker-compose.yml).

FROM python:3.12-slim

# PYTHONUNBUFFERED — вывод идёт в терминал сразу, без буферизации;
#                    без него меню появлялось бы с задержкой.
# PYTHONDONTWRITEBYTECODE — не засорять смонтированные каталоги файлами .pyc.
# TERM — подсказка библиотеке rich, что доступен цветной вывод.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TERM=xterm-256color

# fonts-dejavu-core — TTF-шрифт с кириллицей: без него в PDF-отчёте
#   вместо русских букв печатались бы пустые прямоугольники. Путь
#   /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf уже перечислен
#   в config.PDF_FONT_CANDIDATES.
# ca-certificates — корневые сертификаты для проверки TLS при парсинге.
# tzdata — база часовых поясов: без неё переменная TZ игнорируется и все
#   отметки времени (логи, поле parsed_at, дата в отчёте) идут по UTC.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости ставим отдельным слоем: пока requirements.txt не менялся,
# при пересборке образа этот слой берётся из кеша.
COPY requirements.txt ./

# PyTorch ставим первым и из индекса CPU-сборок. На x86_64 обычный пакет
# torch тянет за собой библиотеки CUDA (несколько гигабайт), которые здесь
# бесполезны — расчёты идут на процессоре. На платформах, где CPU-индекс
# сборки не публикует (например, linux/arm64), происходит откат
# на стандартный индекс PyPI — там для этих платформ и так CPU-сборка.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    || pip install torch==2.13.0

RUN pip install -r requirements.txt

# Исходный код проекта
COPY . .

# Каталоги, которые монтируются томами. Создаём заранее, чтобы права
# были корректными даже при первом запуске без существующих папок на хосте.
RUN mkdir -p /app/data /app/logs /app/reports/output \
    && chmod +x /app/docker/entrypoint.sh

# Точка входа сама наполняет базу при первом запуске, затем передаёт
# управление приложению. Аргументы команды прокидываются в main.py.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD []
