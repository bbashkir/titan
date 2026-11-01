# -*- coding: utf-8 -*-
"""
Конфигурация проекта «Титановые сплавы».

Здесь собраны все настраиваемые параметры системы: пути к файлам,
параметры парсинга, настройки БД, гиперпараметры моделей машинного обучения
и параметры генерации PDF-отчётов.

Менять поведение системы следует именно здесь, а не в коде модулей.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# ПУТИ
# --------------------------------------------------------------------------

# Корневая папка проекта (папка, в которой лежит этот файл)
BASE_DIR = Path(__file__).resolve().parent

# Папка с данными (SQLite-база, сохранённые модели)
DATA_DIR = BASE_DIR / "data"

# Папка для логов
LOGS_DIR = BASE_DIR / "logs"

# Папка для сохранения PDF-отчётов
REPORTS_DIR = BASE_DIR / "reports" / "output"

# Файл базы данных SQLite
DB_PATH = DATA_DIR / "alloys.db"

# Строка подключения SQLAlchemy (SQLite, локальный файл)
DB_URL = f"sqlite:///{DB_PATH}"

# Файл лога парсера
PARSER_LOG = LOGS_DIR / "parser.log"

# Файлы сохранённых обученных моделей
VAE_MODEL_PATH = DATA_DIR / "vae_model.pt"
PROPERTY_MODEL_PATH = DATA_DIR / "property_models.joblib"

# Создаём все нужные каталоги при импорте конфига,
# чтобы дальше в коде не проверять их существование.
for _directory in (DATA_DIR, LOGS_DIR, REPORTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# ПАРСИНГ
# --------------------------------------------------------------------------

# User-Agent, которым представляется парсер.
# Указываем контакт — это требование хорошего тона при работе с чужими сайтами.
USER_AGENT = (
    "TitaniumAlloysResearchBot/1.0 (academic research; "
    "contact: researcher@example.edu)"
)

# Таймаут HTTP-запроса, секунд
REQUEST_TIMEOUT = 20

# Пауза между запросами к одному и тому же домену, секунд.
# Нужна, чтобы не создавать нагрузку на сайт-источник.
REQUEST_DELAY = 3.0

# Количество повторных попыток при сетевой ошибке
MAX_RETRIES = 3

# Проверять ли robots.txt перед обращением к странице.
# ВАЖНО: отключать не рекомендуется — многие источники (MatWeb, Total Materia)
# прямо запрещают автоматический сбор данных в своих правилах.
RESPECT_ROBOTS_TXT = True

# Список источников данных.
#   key      — внутренний идентификатор источника
#   enabled  — включён ли источник в цикл парсинга
#   base_url — базовый адрес
#   lang     — язык источника
SOURCES = {
    # Справочный набор проверенных данных (ГОСТ 19807-91, ОСТ, ASM Handbook).
    # Работает офлайн и всегда доступен — база наполняется даже без интернета.
    "reference": {
        "enabled": True,
        "base_url": "local://reference",
        "lang": "ru/en",
        "title": "Справочные данные (ГОСТ 19807-91, ОСТ 1 90013-81, ASM Handbook)",
    },
    # ASM Aerospace Specification Metals — статический HTML, разбирается надёжно.
    #
    # Выключен: на момент разработки сайт отдаёт сертификат Cloudflare Origin CA,
    # который не подписан ни одним публичным удостоверяющим центром, поэтому
    # проверка TLS не проходит ни у одного клиента (по HTTP идёт редирект на
    # HTTPS). Отключать проверку сертификата нельзя — это допустило бы подмену
    # данных. Парсер источника готов: как только сертификат исправят, достаточно
    # поставить здесь True.
    "asm_aerospace": {
        "enabled": False,
        "base_url": "https://asm.matweb.com",
        "lang": "en",
        "title": "ASM Aerospace Specification Metals",
    },
    # Semantic Scholar — официальный API открытых публикаций.
    # Требует бесплатного ключа (см. SEMANTIC_SCHOLAR_API_KEY ниже):
    # без ключа сервис отвечает 403. Включайте после получения ключа.
    "semantic_scholar": {
        "enabled": True,
        "base_url": "https://api.semanticscholar.org/graph/v1",
        "lang": "en",
        "title": "Semantic Scholar (открытые публикации)",
    },
    # MatWeb — данные защищены условиями использования, автоматический сбор
    # запрещён robots.txt. Источник оставлен выключенным намеренно.
    "matweb": {
        "enabled": False,
        "base_url": "https://www.matweb.com",
        "lang": "en",
        "title": "MatWeb",
    },
    # Total Materia — доступ только по платной подписке, парсинг запрещён.
    "total_materia": {
        "enabled": False,
        "base_url": "https://www.totalmateria.com",
        "lang": "en",
        "title": "Total Materia",
    },
    # ГОСТ-базы: docs.cntd.ru отдаёт содержимое только авторизованным
    # пользователям и закрывает роботов. Тексты ГОСТов внесены в справочный набор.
    "cntd": {
        "enabled": False,
        "base_url": "https://docs.cntd.ru",
        "lang": "ru",
        "title": "Электронный фонд НТД (docs.cntd.ru)",
    },
    # elibrary.ru требует авторизацию и запрещает роботов.
    "elibrary": {
        "enabled": False,
        "base_url": "https://elibrary.ru",
        "lang": "ru",
        "title": "Научная электронная библиотека eLIBRARY.RU",
    },
    # ResearchGate активно блокирует автоматические обращения (Cloudflare).
    "researchgate": {
        "enabled": False,
        "base_url": "https://www.researchgate.net",
        "lang": "en",
        "title": "ResearchGate",
    },
}

# Поисковые запросы для научных источников (Semantic Scholar)
SCHOLAR_QUERIES = [
    "titanium alloy mechanical properties tensile strength",
    "Ti-6Al-4V microstructure mechanical properties",
    "beta titanium alloy elastic modulus biomedical",
    "титановые сплавы механические свойства",
]

# Максимум статей на один поисковый запрос
SCHOLAR_LIMIT = 20

# Ключ API Semantic Scholar.
# Без ключа сервис отвечает 403 Forbidden — бесплатный ключ выдаётся по заявке
# на https://www.semanticscholar.org/product/api. Ключ можно указать здесь
# либо (предпочтительно) в переменной окружения SEMANTIC_SCHOLAR_API_KEY,
# чтобы он не попадал в репозиторий вместе с кодом.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Список названий сплавов, которые ищем в источниках
TARGET_ALLOYS = [
    "Ti-6Al-4V", "Ti-6Al-4V ELI", "Ti-5Al-2.5Sn", "Ti-3Al-2.5V",
    "Ti-6Al-2Sn-4Zr-2Mo", "Ti-10V-2Fe-3Al", "Ti-15V-3Cr-3Sn-3Al",
    "Ti-13Nb-13Zr", "Ti-6Al-7Nb", "Ti-8Al-1Mo-1V",
    "ВТ1-0", "ВТ5", "ВТ6", "ВТ8", "ВТ14", "ВТ20", "ВТ22", "ОТ4", "ПТ3В",
]


# --------------------------------------------------------------------------
# АВТООБНОВЛЕНИЕ (ПЛАНИРОВЩИК)
# --------------------------------------------------------------------------

# Интервал автоматического перепарсинга.
# Поддерживаются значения: "monthly", "weekly", "daily"
SCHEDULE_INTERVAL = "monthly"

# День месяца и время запуска для режима "monthly"
SCHEDULE_DAY_OF_MONTH = 1
SCHEDULE_HOUR = 3
SCHEDULE_MINUTE = 0


# --------------------------------------------------------------------------
# МАШИННОЕ ОБУЧЕНИЕ
# --------------------------------------------------------------------------

# Химические элементы, учитываемые в векторе состава.
# Порядок важен: он определяет размерность входа моделей.
ELEMENTS = [
    "Ti", "Al", "V", "Mo", "Zr", "Sn", "Cr", "Fe",
    "Nb", "Ni", "Mn", "Si", "Pd", "Ta", "Cu", "O",
]

# Числовые свойства, с которыми работают модели
NUMERIC_PROPERTIES = [
    "density",
    "melting_point",
    "tensile_strength",
    "yield_strength",
    "elastic_modulus",
    "hardness_brinell",
    "thermal_conductivity",
    "fatigue_strength",
]

# Человекочитаемые названия свойств и единицы измерения (для вывода)
PROPERTY_LABELS = {
    "density": ("Плотность", "г/см³"),
    "melting_point": ("Температура плавления", "°C"),
    "tensile_strength": ("Предел прочности", "МПа"),
    "yield_strength": ("Предел текучести", "МПа"),
    "elastic_modulus": ("Модуль упругости", "ГПа"),
    "hardness_brinell": ("Твёрдость по Бринеллю", "HB"),
    "thermal_conductivity": ("Теплопроводность", "Вт/(м·К)"),
    "fatigue_strength": ("Усталостная прочность", "МПа"),
    "corrosion_resistance": ("Коррозионная стойкость", ""),
    "application": ("Область применения", ""),
}

# Количество соседей для модели поиска похожих сплавов (kNN)
KNN_NEIGHBORS = 15

# Сколько существующих сплавов показывать в БЛОКЕ А
TOP_SIMILAR = 5

# Сколько сгенерированных сплавов показывать в БЛОКЕ Б
TOP_GENERATED = 3

# --- Гиперпараметры вариационного автокодировщика (VAE) ---

VAE_LATENT_DIM = 4        # размерность латентного пространства
VAE_HIDDEN_DIM = 64       # размер скрытого слоя
VAE_EPOCHS = 400          # число эпох обучения
VAE_LEARNING_RATE = 1e-3  # скорость обучения
VAE_BATCH_SIZE = 64       # размер батча
VAE_KL_WEIGHT = 0.02      # вес KL-дивергенции в функции потерь (beta-VAE)
VAE_AUGMENT_COPIES = 40   # сколько зашумлённых копий делать из каждой записи
VAE_AUGMENT_NOISE = 0.004 # амплитуда шума при аугментации (в долях, т.е. 0.4 %)

# Параметры генерации новых составов
GEN_SAMPLES = 4000        # сколько кандидатов сгенерировать перед отбором
GEN_TEMPERATURE = 1.15    # «температура» выборки из латентного пространства
GEN_MIN_TI = 50.0         # минимальная доля титана, % (требование задачи)
GEN_MIN_ELEMENT = 0.15    # элементы с долей меньше — отбрасываются, %
GEN_NOVELTY_THRESHOLD = 2.0  # минимальное отличие от сплава из БД, % (норма L1)

# Допуск при оценке соответствия запросу.
# Если значение вышло за границу диапазона более чем на TOLERANCE_RATIO
# от ширины запроса — соответствие по этому параметру считается нулевым.
TOLERANCE_RATIO = 0.5


# --------------------------------------------------------------------------
# ОТЧЁТЫ PDF
# --------------------------------------------------------------------------

# Возможные расположения TTF-шрифта с поддержкой кириллицы.
# Проверяются по порядку; берётся первый существующий.
PDF_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\DejaVuSans.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Жирные начертания шрифтов (в том же порядке, что и выше — по возможности)
PDF_FONT_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Шаблон имени файла отчёта
PDF_FILENAME_TEMPLATE = "titanium_report_{timestamp}.pdf"


# --------------------------------------------------------------------------
# ЛОГИРОВАНИЕ
# --------------------------------------------------------------------------

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Максимальный размер файла лога перед ротацией, байт (5 МБ)
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
