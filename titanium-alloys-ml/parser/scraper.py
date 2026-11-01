# -*- coding: utf-8 -*-
"""
Парсер данных о титановых сплавах.

Архитектура модуля:
  * BaseSource            — базовый класс источника (HTTP, robots.txt, паузы);
  * ReferenceSource       — справочный набор проверенных данных (офлайн);
  * AsmAerospaceSource    — карточки материалов ASM Aerospace Specification Metals;
  * SemanticScholarSource — открытый API Semantic Scholar, разбор аннотаций статей;
  * run_parsing()         — оркестратор: обходит источники и пишет в БД.

Принципиальные решения:
  1. Перед обращением к сайту проверяется robots.txt. Источники, которые
     запрещают автоматический сбор (MatWeb, Total Materia, elibrary,
     ResearchGate, docs.cntd.ru), по умолчанию выключены в config.SOURCES —
     их данные внесены в справочный набор с указанием стандарта-источника.
  2. Между запросами выдерживается пауза config.REQUEST_DELAY.
  3. Любая ошибка одного источника не останавливает остальные.
"""

import re
import time
import urllib.robotparser
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config
from database.models import add_alloy, init_db, session_scope
from logger_setup import get_logger
from parser.reference_data import get_reference_records

log = get_logger("parser.scraper")


# ==========================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ РАЗБОРА ТЕКСТА
# ==========================================================================

# Числа вида «4.43», «4,43», «950», «1 200»
_NUMBER = r"(\d{1,4}(?:[  ]\d{3})*(?:[.,]\d+)?)"


def parse_number(raw: Optional[str]) -> Optional[float]:
    """
    Превращает строку в число.
    Понимает запятую как десятичный разделитель, пробелы как разделитель
    разрядов и диапазоны вида «900 - 1100» (берётся середина диапазона).
    Возвращает None, если число извлечь не удалось.
    """
    if raw is None:
        return None
    text = str(raw).replace(" ", " ").strip()
    if not text:
        return None

    # Диапазон значений: берём середину
    range_match = re.search(rf"{_NUMBER}\s*(?:-|–|—|\.\.\.|to)\s*{_NUMBER}", text)
    if range_match:
        low = parse_number(range_match.group(1))
        high = parse_number(range_match.group(2))
        if low is not None and high is not None:
            return round((low + high) / 2, 3)

    single = re.search(_NUMBER, text)
    if not single:
        return None
    cleaned = single.group(1).replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


# Ключевые слова свойств на русском и английском.
# Порядок важен: более специфичные шаблоны идут раньше общих
# (например «предел текучести» до «предел прочности»).
PROPERTY_PATTERNS = [
    ("yield_strength", [
        r"предел\s+текучести", r"условный\s+предел\s+текучести",
        r"yield\s+strength", r"proof\s+stress", r"σ0[.,]2", r"sigma\s*0[.,]2",
    ]),
    ("tensile_strength", [
        r"предел\s+прочности", r"временное\s+сопротивление",
        r"tensile\s+strength", r"ultimate\s+tensile", r"\bUTS\b", r"σв\b",
    ]),
    ("fatigue_strength", [
        r"предел\s+выносливости", r"усталостн\w*\s+прочност",
        r"fatigue\s+strength", r"fatigue\s+limit", r"endurance\s+limit",
    ]),
    ("elastic_modulus", [
        r"модул\w*\s+упругости", r"модул\w*\s+Юнга",
        r"elastic\s+modulus", r"young'?s\s+modulus", r"modulus\s+of\s+elasticity",
    ]),
    ("hardness_brinell", [
        r"твёрдост\w*\s+по\s+Бринел\w*", r"твердост\w*\s+по\s+Бринел\w*",
        r"hardness,?\s+brinell", r"brinell\s+hardness", r"\bHB\b",
    ]),
    ("density", [
        r"плотност\w*", r"\bdensity\b",
    ]),
    ("melting_point", [
        r"температур\w*\s+плавления", r"melting\s+point", r"melting\s+temperature",
        r"\bliquidus\b",
    ]),
    ("thermal_conductivity", [
        r"теплопроводност\w*", r"thermal\s+conductivity",
    ]),
]

# Допустимые физические диапазоны — отсекают заведомо мусорные значения,
# случайно выхваченные из текста.
PROPERTY_RANGES = {
    "density": (3.0, 8.0),
    "melting_point": (1200.0, 2200.0),
    "tensile_strength": (100.0, 2500.0),
    "yield_strength": (80.0, 2400.0),
    "elastic_modulus": (30.0, 250.0),
    "hardness_brinell": (60.0, 600.0),
    "thermal_conductivity": (1.0, 60.0),
    "fatigue_strength": (50.0, 1200.0),
}


def normalize_property_value(prop: str, value: Optional[float],
                             unit_hint: str = "") -> Optional[float]:
    """
    Приводит значение к единицам измерения проекта и отбраковывает
    физически невозможные величины.

    unit_hint — фрагмент исходного текста рядом с числом; по нему
    распознаются ГПа/МПа, кг/м³ и т. п.
    """
    if value is None:
        return None

    hint = unit_hint.lower()

    # Плотность: если пришло в кг/м³ — переводим в г/см³
    if prop == "density" and value > 100:
        value = value / 1000.0

    # Прочность: если указано в ГПа — переводим в МПа
    if prop in ("tensile_strength", "yield_strength", "fatigue_strength"):
        if "гпа" in hint or "gpa" in hint:
            value = value * 1000.0
        elif "ksi" in hint:
            value = value * 6.895

    # Модуль упругости: если указано в МПа — переводим в ГПа
    if prop == "elastic_modulus":
        if "мпа" in hint or "mpa" in hint or value > 1000:
            value = value / 1000.0
        elif "msi" in hint:
            value = value * 6.895

    # Температура: Кельвины -> Цельсии.
    # У титановых сплавов температура плавления не превышает ~1800 °C,
    # поэтому большее значение однозначно указывает на шкалу Кельвина.
    if prop == "melting_point" and (hint.strip() in ("k", "kelvin") or value > 1900):
        value = value - 273.15

    low, high = PROPERTY_RANGES.get(prop, (float("-inf"), float("inf")))
    if not (low <= value <= high):
        return None
    return round(value, 3)


def extract_properties_from_text(text: str) -> Dict[str, float]:
    """
    Ищет в свободном тексте (аннотация статьи, абзац стандарта)
    упоминания свойств с числовыми значениями.

    Работает по принципу «ключевое слово + ближайшее число справа».
    Возвращает словарь найденных свойств.
    """
    found: Dict[str, float] = {}
    flat = re.sub(r"\s+", " ", text)

    for prop, patterns in PROPERTY_PATTERNS:
        if prop in found:
            continue
        for pattern in patterns:
            # Ключевое слово, затем в пределах 60 символов — число с единицами
            regex = rf"{pattern}[^.;\n]{{0,60}}?{_NUMBER}\s*([А-Яа-яA-Za-z°/·²³%.]{{0,12}})"
            match = re.search(regex, flat, flags=re.IGNORECASE)
            if not match:
                continue
            value = normalize_property_value(
                prop, parse_number(match.group(1)), match.group(2) or ""
            )
            if value is not None:
                found[prop] = value
                break
    return found


# Химический состав из названия сплава: Ti-6Al-4V -> {Al: 6, V: 4, Ti: остаток}
_COMPOSITION_TOKEN = re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*([A-Z][a-z]?)")


def parse_composition_from_name(name: str) -> Optional[Dict[str, float]]:
    """
    Восстанавливает номинальный состав из обозначения сплава
    международного типа: «Ti-6Al-4V», «Ti-15V-3Cr-3Sn-3Al».

    Титан считается основой: его доля = 100 % минус сумма легирующих.
    Для отечественных марок (ВТ6, ОТ4) обозначение состав не кодирует,
    поэтому функция вернёт None.
    """
    if not name or not re.match(r"^\s*Ti\b", name, flags=re.IGNORECASE):
        return None

    composition: Dict[str, float] = {}
    for value_str, element in _COMPOSITION_TOKEN.findall(name):
        if element not in config.ELEMENTS or element == "Ti":
            continue
        value = parse_number(value_str)
        if value is not None and 0 < value < 60:
            composition[element] = value

    if not composition:
        return None

    total_alloying = sum(composition.values())
    if total_alloying >= 50:  # титан перестал бы быть основой — состав недостоверен
        return None
    composition["Ti"] = round(100.0 - total_alloying, 2)
    return composition


def guess_corrosion_resistance(text: str) -> Optional[str]:
    """
    Грубая оценка коррозионной стойкости по тексту описания.
    Возвращает 'high' / 'medium' / 'low' или None, если признаков не найдено.
    """
    lowered = text.lower()
    high_markers = [
        "excellent corrosion", "outstanding corrosion", "highly corrosion resistant",
        "высокая коррозионная стойкость", "отличная коррозионная стойкость",
        "стойкость в морской воде", "seawater resistance",
    ]
    medium_markers = [
        "good corrosion", "moderate corrosion",
        "удовлетворительная коррозионная стойкость", "хорошая коррозионная стойкость",
    ]
    low_markers = [
        "poor corrosion", "susceptible to corrosion",
        "низкая коррозионная стойкость", "склонен к коррозии",
    ]
    if any(marker in lowered for marker in high_markers):
        return "high"
    if any(marker in lowered for marker in medium_markers):
        return "medium"
    if any(marker in lowered for marker in low_markers):
        return "low"
    return None


# ==========================================================================
# БАЗОВЫЙ КЛАСС ИСТОЧНИКА
# ==========================================================================

class BaseSource:
    """
    Базовый источник данных.

    Наследники обязаны реализовать collect() — метод, возвращающий список
    словарей-записей в формате таблицы alloys.
    """

    key = "base"

    # Проверять ли robots.txt перед запросом.
    #
    # Для сайтов, страницы которых мы разбираем как HTML, — обязательно.
    # Для документированных публичных REST API проверка отключается: robots.txt
    # регулирует поведение поисковых краулеров, а не обращения к API, и многие
    # API-хосты вообще отдают на /robots.txt код 403. Стандартный
    # urllib.robotparser трактует 403 как «запрещено всё», из-за чего
    # легитимный источник оказался бы недоступен.
    check_robots = True

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.USER_AGENT,
            "Accept-Language": "ru,en;q=0.9",
        })
        # Кеш разобранных robots.txt по доменам
        self._robots_cache: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self._last_request_time = 0.0

    # ------------------------------------------------------------- robots
    def _get_robots(self, url: str) -> Optional[urllib.robotparser.RobotFileParser]:
        """Загружает и кеширует robots.txt для домена указанного URL."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._robots_cache:
            return self._robots_cache[origin]

        robots = urllib.robotparser.RobotFileParser()
        robots.set_url(urljoin(origin, "/robots.txt"))
        try:
            robots.read()
        except Exception as exc:  # сеть недоступна или robots.txt отсутствует
            log.debug("Не удалось прочитать robots.txt для %s: %s", origin, exc)
            robots = None
        self._robots_cache[origin] = robots
        return robots

    def is_allowed(self, url: str) -> bool:
        """Разрешает ли robots.txt обращение к этому URL нашему боту."""
        if not config.RESPECT_ROBOTS_TXT or not self.check_robots:
            return True
        robots = self._get_robots(url)
        if robots is None:
            # robots.txt недоступен — по умолчанию считаем обращение допустимым
            return True
        return robots.can_fetch(config.USER_AGENT, url)

    # ---------------------------------------------------------------- HTTP
    def _throttle(self) -> None:
        """Выдерживает паузу между запросами."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < config.REQUEST_DELAY:
            time.sleep(config.REQUEST_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    def fetch(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """
        Выполняет GET-запрос с проверкой robots.txt, паузами и повторами.
        Возвращает Response либо None, если получить страницу не удалось.
        """
        if not self.is_allowed(url):
            log.warning("[%s] robots.txt запрещает обращение к %s — источник пропущен",
                        self.key, url)
            return None

        for attempt in range(1, config.MAX_RETRIES + 1):
            self._throttle()
            try:
                response = self.session.get(
                    url, params=params, timeout=config.REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    return response
                if response.status_code in (429, 503):
                    # Сервер просит снизить темп — ждём дольше обычного
                    wait = config.REQUEST_DELAY * attempt * 3
                    log.warning("[%s] %s -> HTTP %s, пауза %.0f с",
                                self.key, url, response.status_code, wait)
                    time.sleep(wait)
                    continue
                log.warning("[%s] %s -> HTTP %s", self.key, url, response.status_code)
                return None
            except requests.exceptions.SSLError as exc:
                # Ошибка проверки сертификата не является временной: повторы
                # ничего не изменят. Отключать проверку нельзя — это открыло бы
                # возможность подмены данных источника.
                log.error(
                    "[%s] Не пройдена проверка TLS-сертификата %s: %s. "
                    "Источник пропущен.",
                    self.key, url, str(exc)[:200],
                )
                return None
            except requests.RequestException as exc:
                log.warning("[%s] Ошибка запроса %s (попытка %d/%d): %s",
                            self.key, url, attempt, config.MAX_RETRIES, exc)
                time.sleep(config.REQUEST_DELAY * attempt)
        return None

    # -------------------------------------------------------------- сбор
    def collect(self) -> List[Dict]:
        """Собирает записи из источника. Переопределяется в наследниках."""
        raise NotImplementedError


# ==========================================================================
# ИСТОЧНИК 1. СПРАВОЧНЫЕ ДАННЫЕ (ГОСТ / ОСТ / ASM Handbook)
# ==========================================================================

class ReferenceSource(BaseSource):
    """
    Проверенные справочные данные, внесённые в проект вручную из стандартов
    и справочников. Работает без сети и гарантирует, что база не пустая
    даже при недоступности внешних сайтов.
    """

    key = "reference"

    def collect(self) -> List[Dict]:
        records = get_reference_records()
        log.info("[%s] Подготовлено записей: %d", self.key, len(records))
        return records


# ==========================================================================
# ИСТОЧНИК 2. ASM AEROSPACE SPECIFICATION METALS
# ==========================================================================

class AsmAerospaceSource(BaseSource):
    """
    Разбор карточек материалов ASM Aerospace Specification Metals.

    Страница материала — статический HTML с таблицей «свойство | метрическое
    значение | английское значение». Идентификатор материала (bassnum)
    задаётся в словаре MATERIALS.
    """

    key = "asm_aerospace"

    # Известные идентификаторы карточек титановых сплавов.
    # Список можно расширять: bassnum виден в адресной строке карточки.
    MATERIALS = {
        "Ti-6Al-4V": "MTP641",
        "Ti-6Al-4V ELI": "MTP642",
        "Ti-5Al-2.5Sn": "MTP531",
        "Ti-3Al-2.5V": "MTP321",
    }

    # Соответствие названий строк таблицы полям базы данных
    ROW_MAP = {
        "density": "density",
        "melting point": "melting_point",
        "tensile strength, ultimate": "tensile_strength",
        "tensile strength, yield": "yield_strength",
        "modulus of elasticity": "elastic_modulus",
        "hardness, brinell": "hardness_brinell",
        "thermal conductivity": "thermal_conductivity",
        "fatigue strength": "fatigue_strength",
    }

    def collect(self) -> List[Dict]:
        records: List[Dict] = []
        base = config.SOURCES[self.key]["base_url"]

        for alloy_name, bassnum in self.MATERIALS.items():
            url = f"{base}/search/SpecificMaterial.asp?bassnum={bassnum}"
            response = self.fetch(url)
            if response is None:
                continue
            try:
                record = self._parse_material_page(response.text, alloy_name, url)
            except Exception as exc:
                log.exception("[%s] Ошибка разбора %s: %s", self.key, url, exc)
                continue
            if record:
                records.append(record)
                log.info("[%s] Разобран сплав %s", self.key, alloy_name)

        log.info("[%s] Получено записей: %d", self.key, len(records))
        return records

    def _parse_material_page(self, html: str, alloy_name: str,
                             url: str) -> Optional[Dict]:
        """Извлекает свойства из HTML-карточки материала."""
        soup = BeautifulSoup(html, "html.parser")

        record: Dict = {
            "name": alloy_name,
            "source": url,
            "chemical_composition": parse_composition_from_name(alloy_name),
            "parsed_at": datetime.now(),
        }

        # Свойства лежат в таблицах: первая ячейка — название, вторая — значение
        for row in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if len(cells) < 2:
                continue
            label = cells[0].lower().strip()
            value_text = cells[1]

            for pattern, field in self.ROW_MAP.items():
                if label.startswith(pattern):
                    value = normalize_property_value(
                        field, parse_number(value_text), value_text
                    )
                    if value is not None and record.get(field) is None:
                        record[field] = value
                    break

        # Химический состав из таблицы «Component | Wt. %», если он там есть
        composition = self._parse_composition_table(soup)
        if composition:
            record["chemical_composition"] = composition

        # Область применения и коррозионная стойкость — из текстовых блоков
        page_text = soup.get_text(" ", strip=True)
        record["corrosion_resistance"] = guess_corrosion_resistance(page_text)
        applications = re.search(
            r"(?:Applications?|Typical uses?)\s*[:\-]\s*(.{20,400}?)(?:\.|$)",
            page_text, flags=re.IGNORECASE,
        )
        if applications:
            record["application"] = applications.group(1).strip()

        # Запись имеет смысл только если найдено хотя бы одно свойство
        has_data = any(record.get(prop) is not None for prop in config.NUMERIC_PROPERTIES)
        return record if has_data else None

    @staticmethod
    def _parse_composition_table(soup: BeautifulSoup) -> Optional[Dict[str, float]]:
        """Ищет таблицу химического состава вида «Component | Wt. %»."""
        composition: Dict[str, float] = {}
        for row in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if len(cells) < 2:
                continue
            # Первая ячейка — «Aluminum, Al» либо просто «Al»
            symbol_match = re.search(r"\b([A-Z][a-z]?)\b\s*$", cells[0].strip(" ,"))
            if not symbol_match:
                continue
            symbol = symbol_match.group(1)
            if symbol not in config.ELEMENTS:
                continue
            value = parse_number(cells[1])
            if value is not None and 0 <= value <= 100:
                composition[symbol] = value
        return composition or None


# ==========================================================================
# ИСТОЧНИК 3. SEMANTIC SCHOLAR (открытые научные публикации)
# ==========================================================================

class SemanticScholarSource(BaseSource):
    """
    Сбор данных из аннотаций открытых научных статей через официальный
    API Semantic Scholar (ключ не требуется).

    Из аннотации извлекаются числовые значения свойств и, если в заголовке
    или тексте встречается известное обозначение сплава, формируется запись.
    Такие данные помечаются ссылкой на статью в поле source.
    """

    key = "semantic_scholar"

    # Это публичный REST API, а не сайт для краулинга: robots.txt здесь
    # не применяется (хост отдаёт на него 403). Темп запросов ограничиваем
    # паузой config.REQUEST_DELAY, что укладывается в лимиты API без ключа.
    check_robots = False

    # Обозначения сплавов, которые ищем в тексте статей
    ALLOY_PATTERNS = [
        r"Ti[-\s]?6Al[-\s]?4V\s*ELI", r"Ti[-\s]?6Al[-\s]?4V",
        r"Ti[-\s]?6Al[-\s]?7Nb", r"Ti[-\s]?13Nb[-\s]?13Zr",
        r"Ti[-\s]?5Al[-\s]?2\.5Sn", r"Ti[-\s]?3Al[-\s]?2\.5V",
        r"Ti[-\s]?10V[-\s]?2Fe[-\s]?3Al", r"Ti[-\s]?15V[-\s]?3Cr[-\s]?3Sn[-\s]?3Al",
        r"Ti[-\s]?6Al[-\s]?2Sn[-\s]?4Zr[-\s]?2Mo", r"Ti[-\s]?15Mo",
        r"ВТ\d+[-\dА-Я]*", r"ОТ4[-\d]*", r"ПТ[-\d]?[А-Я]?\w*",
    ]

    def __init__(self):
        super().__init__()
        # Ключ передаётся заголовком x-api-key. Без него сервис отвечает 403.
        if config.SEMANTIC_SCHOLAR_API_KEY:
            self.session.headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

    def collect(self) -> List[Dict]:
        records: List[Dict] = []
        base = config.SOURCES[self.key]["base_url"]
        seen_papers = set()

        if not config.SEMANTIC_SCHOLAR_API_KEY:
            log.warning(
                "[%s] Не задан ключ API (config.SEMANTIC_SCHOLAR_API_KEY или "
                "переменная окружения SEMANTIC_SCHOLAR_API_KEY). Сервис отвечает "
                "403 Forbidden на запросы без ключа — источник пропущен. "
                "Бесплатный ключ: https://www.semanticscholar.org/product/api",
                self.key,
            )
            return []

        for query in config.SCHOLAR_QUERIES:
            url = f"{base}/paper/search"
            params = {
                "query": query,
                "limit": config.SCHOLAR_LIMIT,
                "fields": "title,abstract,year,externalIds,openAccessPdf,url,isOpenAccess",
            }
            response = self.fetch(url, params=params)
            if response is None:
                continue
            try:
                payload = response.json()
            except ValueError:
                log.warning("[%s] Некорректный JSON в ответе на запрос %r", self.key, query)
                continue

            for paper in payload.get("data", []):
                paper_id = paper.get("paperId") or paper.get("url")
                if not paper_id or paper_id in seen_papers:
                    continue
                seen_papers.add(paper_id)

                # Работаем только с открытыми публикациями
                if not paper.get("isOpenAccess") and not paper.get("openAccessPdf"):
                    continue

                records.extend(self._parse_paper(paper))

        log.info("[%s] Получено записей: %d", self.key, len(records))
        return records

    def _parse_paper(self, paper: Dict) -> List[Dict]:
        """Формирует записи по одной статье (по одной на каждый упомянутый сплав)."""
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        text = f"{title}. {abstract}"
        if len(text) < 60:
            return []

        properties = extract_properties_from_text(text)
        if not properties:
            return []

        # Ссылка на статью: DOI предпочтительнее, иначе страница Semantic Scholar
        doi = (paper.get("externalIds") or {}).get("DOI")
        source_url = f"https://doi.org/{doi}" if doi else paper.get("url", "")
        if not source_url:
            return []

        records = []
        for alloy_name in self._find_alloy_names(text):
            record = {
                "name": alloy_name,
                "source": source_url,
                "chemical_composition": parse_composition_from_name(alloy_name),
                "corrosion_resistance": guess_corrosion_resistance(text),
                "application": self._extract_application(text),
                "parsed_at": datetime.now(),
            }
            record.update(properties)
            records.append(record)
        return records

    def _find_alloy_names(self, text: str) -> List[str]:
        """Находит в тексте обозначения сплавов и приводит их к единому виду."""
        found = []
        for pattern in self.ALLOY_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                # Пробелы внутри обозначения заменяем дефисами: «Ti 6Al 4V» -> «Ti-6Al-4V»
                normalized = re.sub(r"\s+", "-", match.group(0).strip())
                if normalized not in found:
                    found.append(normalized)
        # Больше двух сплавов на статью не берём: выше риск ошибочной привязки
        return found[:2]

    @staticmethod
    def _extract_application(text: str) -> Optional[str]:
        """Пытается определить область применения по ключевым словам."""
        areas = {
            "авиация и аэрокосмическая техника": [
                "aerospace", "aircraft", "aeroengine", "turbine", "авиац", "аэрокосм",
            ],
            "медицина и имплантология": [
                "biomedical", "implant", "orthopaedic", "orthopedic", "dental",
                "biocompatib", "медиц", "имплант",
            ],
            "судостроение и морская техника": [
                "marine", "seawater", "naval", "судострое", "морск",
            ],
            "химическая промышленность": [
                "chemical processing", "corrosive media", "химическ",
            ],
            "аддитивные технологии": [
                "additive manufacturing", "selective laser melting", "3d print",
                "аддитивн",
            ],
        }
        lowered = text.lower()
        matched = [area for area, keys in areas.items()
                   if any(key in lowered for key in keys)]
        return ", ".join(matched) if matched else None


# ==========================================================================
# ОРКЕСТРАТОР ПАРСИНГА
# ==========================================================================

# Реестр доступных источников: ключ конфига -> класс
SOURCE_CLASSES = {
    "reference": ReferenceSource,
    "asm_aerospace": AsmAerospaceSource,
    "semantic_scholar": SemanticScholarSource,
}


def run_parsing(source_keys: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Полный цикл парсинга: обход источников, запись новых данных в БД.

    source_keys — какие источники обрабатывать; по умолчанию берутся все
    включённые в config.SOURCES.

    Возвращает статистику: сколько записей добавлено, сколько пропущено
    как дубли, сколько источников отработало с ошибкой.
    """
    init_db()

    if source_keys is None:
        source_keys = [
            key for key, meta in config.SOURCES.items()
            if meta.get("enabled") and key in SOURCE_CLASSES
        ]

    stats = {"added": 0, "variant": 0, "duplicate": 0, "failed_sources": 0,
             "collected": 0}

    log.info("=" * 70)
    log.info("Запуск парсинга. Источники: %s", ", ".join(source_keys) or "нет")

    for key in source_keys:
        source_class = SOURCE_CLASSES.get(key)
        if source_class is None:
            log.warning("Неизвестный источник %r — пропущен", key)
            continue

        log.info("-" * 70)
        log.info("Источник: %s", config.SOURCES.get(key, {}).get("title", key))

        try:
            records = source_class().collect()
        except Exception as exc:
            log.exception("Источник %s завершился ошибкой: %s", key, exc)
            stats["failed_sources"] += 1
            continue

        stats["collected"] += len(records)
        source_stats = {"added": 0, "variant": 0, "duplicate": 0}

        # Каждую запись пишем в отдельной транзакции: ошибка на одной
        # записи не должна отменять успешно сохранённые.
        for record in records:
            try:
                with session_scope() as session:
                    status = add_alloy(session, record)
                source_stats[status] += 1
                stats[status] += 1
            except Exception as exc:
                log.warning("Не удалось сохранить запись %r из %r: %s",
                            record.get("name"), key, exc)

        log.info(
            "Источник %s: добавлено новых — %d, новых вариантов состава — %d, "
            "пропущено дублей — %d",
            key, source_stats["added"], source_stats["variant"],
            source_stats["duplicate"],
        )

    log.info("=" * 70)
    log.info(
        "ИТОГ ПАРСИНГА: собрано %d записей | добавлено новых: %d | "
        "новых вариантов: %d | пропущено дублей: %d | источников с ошибкой: %d",
        stats["collected"], stats["added"], stats["variant"],
        stats["duplicate"], stats["failed_sources"],
    )
    return stats


if __name__ == "__main__":
    # Ручной запуск парсинга: python -m parser.scraper
    from logger_setup import enable_console_info

    enable_console_info()
    run_parsing()
