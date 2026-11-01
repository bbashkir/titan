# -*- coding: utf-8 -*-
"""
Генеративная модель новых титановых сплавов.

Основа — вариационный автокодировщик (VAE) на PyTorch, обученный на
химических составах из базы данных.

Как обеспечивается физико-химическая корректность:
  * выход декодера проходит через softmax, поэтому сумма долей элементов
    всегда равна ровно 100 %;
  * составы с долей титана ниже config.GEN_MIN_TI (50 %) отбраковываются —
    титан обязан оставаться основой;
  * элементы с долей менее config.GEN_MIN_ELEMENT отбрасываются, остаток
    перенормируется (в реальном сплаве следовые количества не указывают);
  * кандидаты, слишком похожие на существующие сплавы (норма L1 меньше
    config.GEN_NOVELTY_THRESHOLD), отбрасываются — модель должна предлагать
    именно новые комбинации, а не повторять базу.

Свойства сгенерированных составов предсказываются регрессорами Random Forest
(класс PropertyPredictor), обученными на тех же данных.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

import config
from logger_setup import get_logger
from ml.features import (
    MatchResult,
    UserQuery,
    build_explanation,
    composition_to_vector,
    element_matrix,
    evaluate_match,
    fmt,
    score_numeric,
    vector_to_composition,
)
from ml.similarity_model import PropertyPredictor

log = get_logger("ml.generative_model")

# Разумная точность вывода предсказанных свойств.
# Регрессия выдаёт значения с машинной точностью, но заявлять «1610.79 °C»
# для прогноза бессмысленно — округляем до практически значимых разрядов.
PROPERTY_ROUNDING = {
    "density": 2,
    "thermal_conductivity": 1,
    "elastic_modulus": 1,
    "melting_point": 0,
    "tensile_strength": 0,
    "yield_strength": 0,
    "hardness_brinell": 0,
    "fatigue_strength": 0,
}

# Металловедческая роль легирующих элементов — используется при построении
# текстового объяснения, почему модель предложила именно такой состав.
ELEMENT_ROLES = {
    "Al": "альфа-стабилизатор: повышает прочность и жаропрочность, снижает плотность",
    "V": "бета-стабилизатор: улучшает пластичность и прокаливаемость",
    "Mo": "бета-стабилизатор: повышает прочность и стойкость к ползучести",
    "Cr": "бета-стабилизатор: увеличивает прочность после старения",
    "Fe": "недорогой бета-стабилизатор: повышает прочность, ухудшает свариваемость",
    "Mn": "бета-стабилизатор: упрочняет твёрдый раствор при хорошей штампуемости",
    "Zr": "нейтральный упрочнитель: повышает прочность без потери пластичности",
    "Sn": "нейтральный упрочнитель твёрдого раствора, улучшает жаропрочность",
    "Nb": "бета-стабилизатор: снижает модуль упругости, повышает биосовместимость",
    "Ta": "повышает коррозионную стойкость и биосовместимость",
    "Si": "повышает сопротивление ползучести при высоких температурах",
    "Pd": "резко повышает коррозионную стойкость в восстановительных средах",
    "Ni": "повышает стойкость к щелевой коррозии в горячих хлоридных средах",
    "Cu": "упрочняет при термообработке, улучшает обрабатываемость",
    "O": "внедрённая примесь: повышает прочность, снижает пластичность",
}


# ==========================================================================
# АРХИТЕКТУРА VAE
# ==========================================================================

class CompositionVAE(nn.Module):
    """
    Вариационный автокодировщик для векторов химического состава.

    Вход и выход — доли элементов (сумма = 1). Выход декодера проходит
    через softmax, что гарантирует нормировку состава.
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int = config.VAE_HIDDEN_DIM,
                 latent_dim: int = config.VAE_LATENT_DIM):
        super().__init__()

        # Энкодер: состав -> параметры распределения в латентном пространстве
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Декодер: точка латентного пространства -> состав
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor):
        """Кодирует состав в параметры нормального распределения."""
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Трюк репараметризации: z = mu + sigma * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодирует латентный вектор в состав (softmax гарантирует сумму 1)."""
        return F.softmax(self.decoder(z), dim=-1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstructed: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Функция потерь VAE.

    Складывается из трёх частей:
      * перекрёстная энтропия — согласует «профиль» состава;
      * квадратичная ошибка — следит за точностью конкретных долей
        (важно для малых добавок вроде 0.3 % Si);
      * KL-дивергенция с весом config.VAE_KL_WEIGHT (бета-VAE) — делает
        латентное пространство гладким, чтобы из него можно было
        осмысленно семплировать новые составы.
    """
    eps = 1e-9
    cross_entropy = -(target * torch.log(reconstructed + eps)).sum(dim=1).mean()
    mse = ((reconstructed - target) ** 2).sum(dim=1).mean()
    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    ).mean()
    return cross_entropy + 50.0 * mse + config.VAE_KL_WEIGHT * kl_divergence


# ==========================================================================
# РЕЗУЛЬТАТ ГЕНЕРАЦИИ
# ==========================================================================

@dataclass
class GeneratedAlloy:
    """Сгенерированный сплав вместе с предсказанными свойствами и оценкой."""

    composition: Dict[str, float]
    properties: Dict[str, float]
    corrosion_resistance: Optional[str]
    application: Optional[str]
    match: MatchResult
    analog_name: str = ""
    analog_source: str = ""
    analog_percent: float = 0.0
    analog_distance: float = 0.0
    explanation: str = ""
    design_rationale: str = ""
    advantage: str = ""

    note: str = "Внимание: сплав не существует в БД, является прогнозом модели"

    @property
    def percent(self) -> float:
        return self.match.percent

    def display_name(self) -> str:
        """
        Условное обозначение сгенерированного сплава по международной схеме:
        Ti-6.7Al-3.2V-1.8Mo (титан-основа не указывается в обозначении).
        """
        parts = [
            f"{value:g}{element}"
            for element, value in sorted(
                self.composition.items(), key=lambda kv: -kv[1]
            )
            if element != "Ti" and value >= 0.5
        ]
        return "Ti-" + "-".join(parts) if parts else "Ti (нелегированный)"


# ==========================================================================
# ГЕНЕРАТИВНАЯ МОДЕЛЬ
# ==========================================================================

class GenerativeModel:
    """
    Обучает VAE на составах из БД и генерирует новые сплавы,
    ранжируя их по соответствию запросу пользователя.
    """

    def __init__(self, predictor: Optional[PropertyPredictor] = None):
        self.vae: Optional[CompositionVAE] = None
        # Регрессоры свойств: можно передать уже обученные из модели поиска,
        # чтобы не обучать одно и то же дважды.
        self.predictor = predictor or PropertyPredictor()
        self._predictor_is_external = predictor is not None
        # Классификатор коррозионной стойкости по составу
        self.corrosion_model: Optional[RandomForestClassifier] = None
        self.df: Optional[pd.DataFrame] = None
        self.db_matrix: Optional[np.ndarray] = None
        self.is_fitted = False

    # ------------------------------------------------------------- обучение

    def fit(self, df: pd.DataFrame) -> "GenerativeModel":
        """Полное обучение: VAE + предсказатели свойств и стойкости."""
        if df is None or df.empty:
            raise ValueError("Невозможно обучить генеративную модель: база пуста")

        self.df = df.reset_index(drop=True)
        self.db_matrix = element_matrix(self.df)

        # Оставляем только записи с непустым и корректным составом
        valid_mask = self.db_matrix.sum(axis=1) > 50.0
        training_matrix = self.db_matrix[valid_mask]
        if len(training_matrix) < 3:
            raise ValueError(
                "Недостаточно записей с химическим составом для обучения VAE"
            )

        if not self._predictor_is_external:
            self.predictor.fit(self.df)

        self._fit_corrosion_classifier()
        self._train_vae(training_matrix)

        self.is_fitted = True
        return self

    def _fit_corrosion_classifier(self) -> None:
        """Обучает классификатор коррозионной стойкости по химическому составу."""
        labels = self.df["corrosion_resistance"]
        mask = labels.notna().to_numpy() & (labels.fillna("") != "").to_numpy()
        unique_labels = set(labels[mask].tolist())

        if int(mask.sum()) < 5 or len(unique_labels) < 2:
            log.info("Недостаточно размеченных данных для классификатора "
                     "коррозионной стойкости — будет использован ближайший аналог")
            return

        model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        model.fit(self.db_matrix[mask], labels[mask].to_numpy())
        self.corrosion_model = model
        log.info("Классификатор коррозионной стойкости обучен на %d примерах",
                 int(mask.sum()))

    def _augment(self, matrix: np.ndarray) -> np.ndarray:
        """
        Аугментация обучающей выборки.

        Записей в базе немного, поэтому каждая копируется несколько раз
        с небольшим шумом. Это расширяет область определения VAE и
        предотвращает вырождение латентного пространства в точки.
        """
        fractions = matrix / matrix.sum(axis=1, keepdims=True)
        copies = [fractions]

        rng = np.random.default_rng(42)
        for _ in range(config.VAE_AUGMENT_COPIES):
            noise = rng.normal(0.0, config.VAE_AUGMENT_NOISE, size=fractions.shape)
            # Шум добавляем только к присутствующим элементам и к «соседним» —
            # так модель узнаёт о возможности небольших добавок.
            presence = (fractions > 0).astype(float)
            neighbour = rng.random(fractions.shape) < 0.05
            noisy = fractions + noise * np.maximum(presence, neighbour.astype(float))
            noisy = np.clip(noisy, 0.0, None)
            noisy = noisy / noisy.sum(axis=1, keepdims=True)
            copies.append(noisy)

        return np.vstack(copies)

    def _train_vae(self, matrix: np.ndarray) -> None:
        """Обучает вариационный автокодировщик."""
        torch.manual_seed(42)

        data = self._augment(matrix)
        tensor = torch.tensor(data, dtype=torch.float32)
        dataset_size = len(tensor)

        self.vae = CompositionVAE(input_dim=len(config.ELEMENTS))
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=config.VAE_LEARNING_RATE)

        self.vae.train()
        batch_size = min(config.VAE_BATCH_SIZE, dataset_size)
        last_loss = float("nan")

        for epoch in range(config.VAE_EPOCHS):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            batches = 0

            for start in range(0, dataset_size, batch_size):
                batch = tensor[permutation[start:start + batch_size]]
                optimizer.zero_grad()
                reconstructed, mu, logvar = self.vae(batch)
                loss = vae_loss(reconstructed, batch, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1

            last_loss = epoch_loss / max(batches, 1)
            if (epoch + 1) % 100 == 0:
                log.info("VAE, эпоха %d/%d: функция потерь = %.5f",
                         epoch + 1, config.VAE_EPOCHS, last_loss)

        self.vae.eval()
        log.info("VAE обучен: %d эпох, %d обучающих векторов, итоговая потеря %.5f",
                 config.VAE_EPOCHS, dataset_size, last_loss)

    # ------------------------------------------------------------- генерация

    def _sample_candidates(self, n_samples: int) -> np.ndarray:
        """Семплирует составы из латентного пространства VAE (в процентах)."""
        with torch.no_grad():
            z = torch.randn(n_samples, config.VAE_LATENT_DIM) * config.GEN_TEMPERATURE
            decoded = self.vae.decode(z).numpy()
        return decoded * 100.0

    def _is_novel(self, vector: np.ndarray,
                  accepted: List[np.ndarray]) -> bool:
        """
        Проверяет новизну состава: он должен заметно отличаться и от записей
        БД, и от уже отобранных кандидатов (норма L1 в процентных пунктах).
        """
        if self.db_matrix is not None and len(self.db_matrix):
            db_distance = np.abs(self.db_matrix - vector).sum(axis=1).min()
            if db_distance < config.GEN_NOVELTY_THRESHOLD:
                return False
        for other in accepted:
            if np.abs(other - vector).sum() < config.GEN_NOVELTY_THRESHOLD * 2:
                return False
        return True

    def generate(self, query: UserQuery,
                 top_n: int = config.TOP_GENERATED,
                 similarity_model=None) -> List[GeneratedAlloy]:
        """
        Генерирует новые сплавы и возвращает топ-N по соответствию запросу.

        similarity_model — модель поиска похожих; используется, чтобы найти
        ближайший аналог из БД для каждого сгенерированного состава.
        """
        if not self.is_fitted:
            raise RuntimeError("Генеративная модель не обучена — вызовите fit()")

        raw_candidates = self._sample_candidates(config.GEN_SAMPLES)

        accepted_vectors: List[np.ndarray] = []
        results: List[GeneratedAlloy] = []

        for raw in raw_candidates:
            # Приводим к «инженерному» виду: убираем следовые элементы,
            # нормируем сумму ровно к 100 %, округляем до 0.1 %.
            composition = vector_to_composition(raw)
            if not composition:
                continue

            # Требование задачи: титан — основа, не менее 50 %
            if composition.get("Ti", 0.0) < config.GEN_MIN_TI:
                continue

            vector = composition_to_vector(composition)
            if not self._is_novel(vector, accepted_vectors):
                continue

            generated = self._build_result(composition, vector, query, similarity_model)
            accepted_vectors.append(vector)
            results.append(generated)

            # Кандидатов набираем с запасом: отбор идёт по проценту соответствия
            if len(results) >= top_n * 12:
                break

        if not results:
            log.warning("Генеративная модель не смогла построить корректные составы")
            return []

        results.sort(key=lambda item: -item.percent)
        top = results[:top_n]

        log.info("Сгенерировано корректных составов: %d, отобрано: %d",
                 len(results), len(top))
        return top

    def _build_result(self, composition: Dict[str, float], vector: np.ndarray,
                      query: UserQuery, similarity_model) -> GeneratedAlloy:
        """Досчитывает свойства, оценку и объяснения для одного состава."""
        properties = self.predictor.predict_one(vector)
        properties = {
            prop: round(value, PROPERTY_ROUNDING.get(prop, 0))
            for prop, value in properties.items()
        }

        # Ближайший аналог из базы — нужен и для объяснения, и как источник
        # области применения / коррозионной стойкости при отсутствии модели.
        analog = None
        if similarity_model is not None:
            analog = similarity_model.nearest_by_composition(vector)
        elif self.df is not None:
            distances = np.abs(self.db_matrix - vector).sum(axis=1)
            analog = self.df.iloc[int(np.argmin(distances))]

        corrosion = self._predict_corrosion(vector, analog)
        application = self._predict_application(analog)

        match = evaluate_match(
            properties=properties,
            corrosion=corrosion,
            application=application,
            query=query,
        )

        result = GeneratedAlloy(
            composition=composition,
            properties=properties,
            corrosion_resistance=corrosion,
            application=application,
            match=match,
            explanation=build_explanation(match),
        )

        if analog is not None:
            analog_properties = {
                prop: (None if pd.isna(analog[prop]) else float(analog[prop]))
                for prop in config.NUMERIC_PROPERTIES
            }
            analog_match = evaluate_match(
                properties=analog_properties,
                corrosion=analog["corrosion_resistance"],
                application=analog["application"],
                query=query,
            )
            result.analog_name = str(analog["name"])
            result.analog_source = str(analog["source"])
            result.analog_percent = analog_match.percent
            result.analog_distance = float(
                np.abs(composition_to_vector(analog["composition"]) - vector).sum()
            )
            result.advantage = self._compare_with_analog(
                properties, analog_properties, analog, query
            )

        result.design_rationale = self._explain_design(composition)
        return result

    def _predict_corrosion(self, vector: np.ndarray, analog) -> Optional[str]:
        """Предсказывает коррозионную стойкость нового состава."""
        if self.corrosion_model is not None:
            return str(self.corrosion_model.predict(vector.reshape(1, -1))[0])
        if analog is not None:
            return analog["corrosion_resistance"]
        return None

    @staticmethod
    def _predict_application(analog) -> Optional[str]:
        """
        Область применения нового сплава оценивается по ближайшему аналогу:
        близкий состав — близкая ниша применения.
        """
        if analog is None:
            return None
        application = analog["application"]
        if not application:
            return None
        return f"по аналогии с «{analog['name']}»: {application}"

    def _explain_design(self, composition: Dict[str, float]) -> str:
        """
        Объясняет, почему модель предложила именно такой состав:
        перечисляет роль каждого легирующего элемента.
        """
        titanium = composition.get("Ti", 0.0)
        parts = [
            f"Основа — титан {titanium:.1f} %, что сохраняет характерное для "
            f"титановых сплавов сочетание малой плотности и коррозионной стойкости."
        ]

        alloying = sorted(
            ((el, val) for el, val in composition.items() if el != "Ti"),
            key=lambda kv: -kv[1],
        )
        if alloying:
            roles = [
                f"{element} {value:.1f} % — {ELEMENT_ROLES.get(element, 'легирующая добавка')}"
                for element, value in alloying[:5]
            ]
            parts.append("Легирование: " + "; ".join(roles) + ".")

        # Баланс альфа- и бета-стабилизаторов — ключевая характеристика сплава
        alpha = sum(composition.get(el, 0.0) for el in ("Al", "O"))
        beta = sum(composition.get(el, 0.0) for el in ("V", "Mo", "Cr", "Fe", "Nb", "Mn"))
        if beta > alpha * 1.5 and beta > 6:
            parts.append(
                f"Суммарное содержание бета-стабилизаторов ({beta:.1f} %) заметно "
                f"превышает содержание альфа-стабилизаторов ({alpha:.1f} %), поэтому "
                f"состав отнесён к псевдо-бета-сплавам: высокая прочность после "
                f"старения при умеренном модуле упругости."
            )
        elif alpha > beta * 1.5 and alpha > 3:
            parts.append(
                f"Преобладание альфа-стабилизаторов ({alpha:.1f} % против {beta:.1f} %) "
                f"даёт хорошую свариваемость и жаропрочность при меньшей склонности "
                f"к упрочняющей термообработке."
            )
        else:
            parts.append(
                f"Альфа- ({alpha:.1f} %) и бета-стабилизаторы ({beta:.1f} %) "
                f"сбалансированы — состав относится к двухфазным (альфа+бета) "
                f"сплавам, наиболее универсальным по свойствам."
            )
        return " ".join(parts)

    @staticmethod
    def _margin(value: Optional[float], vmin: Optional[float],
                vmax: Optional[float]) -> Optional[float]:
        """
        Запас значения относительно границ запроса: насколько далеко оно
        от ближайшего ограничения внутри допустимой области.

        Нужен, чтобы различать варианты, которые оба формально проходят
        по критерию: больший запас означает более надёжное выполнение
        требования с учётом разброса свойств реального металла.
        """
        if value is None:
            return None
        margins = []
        if vmin is not None:
            margins.append(value - vmin)
        if vmax is not None:
            margins.append(vmax - value)
        return min(margins) if margins else None

    @classmethod
    def _compare_with_analog(cls, generated: Dict[str, float],
                             analog_properties: Dict[str, Optional[float]],
                             analog, query: UserQuery) -> str:
        """
        Формулирует, чем сгенерированный сплав лучше ближайшего аналога
        именно в контексте запроса пользователя.

        Сравнение идёт в два уровня: сначала по оценке соответствия, а если
        оба варианта критерий выполняют, — по запасу относительно границы.
        """
        improvements: List[str] = []
        regressions: List[str] = []

        for prop, (vmin, vmax) in query.ranges.items():
            generated_value_raw = generated.get(prop)
            analog_value_raw = analog_properties.get(prop)

            generated_score, _ = score_numeric(prop, generated_value_raw, vmin, vmax)
            analog_score, _ = score_numeric(prop, analog_value_raw, vmin, vmax)
            if generated_score is None:
                continue

            label, unit = config.PROPERTY_LABELS[prop]
            generated_value = f"{fmt(generated_value_raw)} {unit}".strip()
            analog_value = f"{fmt(analog_value_raw)} {unit}".strip()

            if analog_score is None:
                improvements.append(
                    f"{label}: у аналога значение в источнике отсутствует, "
                    f"модель даёт оценку {generated_value}"
                )
            elif generated_score > analog_score + 0.02:
                improvements.append(
                    f"{label}: {generated_value} против {analog_value} у аналога"
                )
            elif analog_score > generated_score + 0.02:
                regressions.append(
                    f"{label}: {generated_value} против {analog_value} у аналога"
                )
            else:
                # Оба варианта проходят по критерию — сравниваем запас
                generated_margin = cls._margin(generated_value_raw, vmin, vmax)
                analog_margin = cls._margin(analog_value_raw, vmin, vmax)
                if generated_margin is None or analog_margin is None:
                    continue
                difference = generated_margin - analog_margin
                # Значимой считаем разницу больше 2 % от величины свойства
                threshold = max(abs(generated_value_raw) * 0.02, 1e-6)
                if difference > threshold:
                    improvements.append(
                        f"{label}: {generated_value} против {analog_value} у аналога — "
                        f"больший запас относительно заданной границы"
                    )
                elif -difference > threshold:
                    regressions.append(
                        f"{label}: {generated_value} против {analog_value} у аналога — "
                        f"меньший запас относительно заданной границы"
                    )

        analog_name = str(analog["name"]) if analog is not None else "аналог"
        if not improvements and not regressions:
            return (
                f"По заданным критериям состав близок к «{analog_name}», но отличается "
                f"химическим составом, что расширяет пространство поиска."
            )

        text = f"Относительно ближайшего аналога «{analog_name}»: "
        if improvements:
            text += "преимущества — " + "; ".join(improvements) + ". "
        if regressions:
            text += "уступает по — " + "; ".join(regressions) + "."
        return text.strip()

    # ------------------------------------------------- сохранение / загрузка

    def save(self, path=config.VAE_MODEL_PATH) -> None:
        """Сохраняет веса VAE на диск."""
        if self.vae is None:
            return
        torch.save({
            "state_dict": self.vae.state_dict(),
            "input_dim": len(config.ELEMENTS),
            "latent_dim": config.VAE_LATENT_DIM,
            "hidden_dim": config.VAE_HIDDEN_DIM,
        }, str(path))
        log.info("Веса VAE сохранены: %s", path)

    def load(self, path=config.VAE_MODEL_PATH) -> bool:
        """
        Загружает ранее сохранённые веса VAE.
        Возвращает True при успехе. Регрессоры свойств при этом
        всё равно нужно обучить методом fit() — они дешёвые.
        """
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            log.info("Сохранённая модель VAE не загружена (%s)", exc)
            return False

        if checkpoint.get("input_dim") != len(config.ELEMENTS):
            log.warning("Сохранённая модель несовместима с текущим списком элементов")
            return False

        self.vae = CompositionVAE(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint.get("hidden_dim", config.VAE_HIDDEN_DIM),
            latent_dim=checkpoint.get("latent_dim", config.VAE_LATENT_DIM),
        )
        self.vae.load_state_dict(checkpoint["state_dict"])
        self.vae.eval()
        log.info("Веса VAE загружены: %s", path)
        return True
