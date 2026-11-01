# -*- coding: utf-8 -*-
"""
Модель интерполяции — поиск реальных сплавов, наиболее близких к запросу.

Работает в два этапа:

  1. Заполнение пропусков (Random Forest регрессия).
     В базе многие свойства отсутствуют (NULL). Для каждого свойства
     обучается случайный лес, предсказывающий его по химическому составу.
     Полученные значения используются ТОЛЬКО для геометрии поиска —
     в отчёт они не попадают, там по-прежнему выводится прочерк.

  2. Поиск ближайших соседей (kNN).
     Запрос пользователя превращается в точку в пространстве свойств,
     после чего NearestNeighbors отбирает кандидатов, а окончательное
     ранжирование выполняется по проценту соответствия, посчитанному
     на реальных (не восстановленных) значениях.

Такая связка даёт устойчивый результат на небольшой и неполной выборке.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config
from logger_setup import get_logger
from ml.features import (
    MatchResult,
    UserQuery,
    build_explanation,
    element_matrix,
    evaluate_match,
)

log = get_logger("ml.similarity_model")

# Минимальное число размеченных примеров, при котором есть смысл
# обучать регрессор для конкретного свойства
MIN_SAMPLES_FOR_REGRESSION = 6


@dataclass
class SimilarAlloy:
    """Найденный существующий сплав вместе с оценкой соответствия."""

    alloy_id: int
    name: str
    source: str
    composition: Dict[str, float]
    properties: Dict[str, Optional[float]]
    corrosion_resistance: Optional[str]
    application: Optional[str]
    match: MatchResult
    distance: float = 0.0
    explanation: str = ""

    @property
    def percent(self) -> float:
        return self.match.percent


class PropertyPredictor:
    """
    Набор регрессоров «химический состав -> свойство».

    Используется дважды:
      * в модели поиска — для восстановления пропущенных значений;
      * в генеративной модели — для предсказания свойств новых составов.
    """

    def __init__(self):
        # Обученные модели по свойствам
        self.models: Dict[str, RandomForestRegressor] = {}
        # Медианные значения — запасной вариант, если обучить модель не удалось
        self.fallbacks: Dict[str, float] = {}
        # Оценка качества (R² на обучающей выборке) — для диагностики в логе
        self.scores: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "PropertyPredictor":
        """Обучает по одному регрессору на каждое числовое свойство."""
        X_all = element_matrix(df)

        for prop in config.NUMERIC_PROPERTIES:
            values = pd.to_numeric(df[prop], errors="coerce")
            mask = values.notna().to_numpy()
            n_samples = int(mask.sum())

            if n_samples == 0:
                continue

            self.fallbacks[prop] = float(values[mask].median())

            if n_samples < MIN_SAMPLES_FOR_REGRESSION:
                log.info("Свойство %s: только %d значений — используется медиана",
                         prop, n_samples)
                continue

            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=1,
                max_features=0.7,
                random_state=42,
                n_jobs=-1,
            )
            X = X_all[mask]
            y = values[mask].to_numpy(dtype=float)
            model.fit(X, y)

            self.models[prop] = model
            self.scores[prop] = float(model.score(X, y))
            log.info("Свойство %s: обучен Random Forest на %d примерах (R²=%.3f)",
                     prop, n_samples, self.scores[prop])

        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Предсказывает все свойства для матрицы составов X.
        Для свойств без обученной модели возвращает медианное значение.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions: Dict[str, np.ndarray] = {}
        for prop in config.NUMERIC_PROPERTIES:
            if prop in self.models:
                predictions[prop] = self.models[prop].predict(X)
            elif prop in self.fallbacks:
                predictions[prop] = np.full(len(X), self.fallbacks[prop])
        return predictions

    def predict_one(self, composition_vector: np.ndarray) -> Dict[str, float]:
        """Предсказание свойств для одного состава — словарём."""
        predictions = self.predict(composition_vector.reshape(1, -1))
        return {prop: float(values[0]) for prop, values in predictions.items()}


class SimilarityModel:
    """Поиск существующих сплавов, максимально близких к запросу пользователя."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.predictor = PropertyPredictor()
        self.scaler: Optional[StandardScaler] = None
        # Матрица свойств с заполненными пропусками (порядок — config.NUMERIC_PROPERTIES)
        self.filled_matrix: Optional[np.ndarray] = None
        self.is_fitted = False

    # ------------------------------------------------------------- обучение

    def fit(self, df: pd.DataFrame) -> "SimilarityModel":
        """Обучает модель на таблице записей из БД."""
        if df is None or df.empty:
            raise ValueError("Невозможно обучить модель: база данных пуста")

        self.df = df.reset_index(drop=True)
        self.predictor.fit(self.df)
        self.filled_matrix = self._build_filled_matrix()

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.filled_matrix)

        # NearestNeighbors обучаем на полном пространстве свойств;
        # при поиске по подмножеству свойств строим отдельный индекс.
        self._scaled_matrix = scaled
        self.is_fitted = True

        log.info("Модель поиска похожих обучена на %d записях", len(self.df))
        return self

    def _build_filled_matrix(self) -> np.ndarray:
        """
        Матрица свойств размера (n_записей x n_свойств), где NULL заменены
        предсказаниями Random Forest по химическому составу.
        """
        X_comp = element_matrix(self.df)
        predicted = self.predictor.predict(X_comp)

        columns = []
        for prop in config.NUMERIC_PROPERTIES:
            real = pd.to_numeric(self.df[prop], errors="coerce").to_numpy(dtype=float)
            if prop in predicted:
                filled = np.where(np.isnan(real), predicted[prop], real)
            else:
                # Ни одного значения в базе — заполняем нулями,
                # столбец фактически перестаёт влиять на расстояние.
                filled = np.nan_to_num(real, nan=0.0)
            columns.append(filled)
        return np.column_stack(columns)

    # --------------------------------------------------------------- поиск

    def _query_point(self, query: UserQuery) -> Dict[str, float]:
        """
        Переводит запрос в «идеальную» точку в пространстве свойств:
        для двусторонних диапазонов — середина, для односторонних —
        граница с небольшим отступом внутрь допустимой области.
        """
        point: Dict[str, float] = {}
        for prop, (vmin, vmax) in query.ranges.items():
            if vmin is not None and vmax is not None:
                point[prop] = (vmin + vmax) / 2.0
            elif vmin is not None:
                point[prop] = vmin * 1.05
            elif vmax is not None:
                point[prop] = vmax * 0.95
        return point

    def _knn_candidates(self, query: UserQuery) -> List[int]:
        """
        Отбирает индексы записей-кандидатов методом ближайших соседей
        в подпространстве запрошенных свойств.
        """
        point = self._query_point(query)
        n_rows = len(self.df)

        if not point:
            # Числовые критерии не заданы — кандидатами считаем все записи
            return list(range(n_rows))

        indices = [config.NUMERIC_PROPERTIES.index(prop) for prop in point]
        sub_matrix = self._scaled_matrix[:, indices]

        # Точку запроса масштабируем теми же параметрами, что и обучающие данные
        target = np.zeros(len(config.NUMERIC_PROPERTIES))
        for prop, value in point.items():
            target[config.NUMERIC_PROPERTIES.index(prop)] = value
        scaled_target = self.scaler.transform(target.reshape(1, -1))[0][indices]

        n_neighbors = min(config.KNN_NEIGHBORS, n_rows)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(sub_matrix)
        distances, neighbors = knn.kneighbors(scaled_target.reshape(1, -1))

        self._last_distances = {
            int(idx): float(dist)
            for idx, dist in zip(neighbors[0], distances[0])
        }
        return [int(idx) for idx in neighbors[0]]

    def find_similar(self, query: UserQuery,
                     top_n: int = config.TOP_SIMILAR) -> List[SimilarAlloy]:
        """
        Возвращает топ-N существующих сплавов, отсортированных
        по проценту соответствия запросу.
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена — сначала вызовите fit()")

        self._last_distances: Dict[int, float] = {}
        candidate_indices = self._knn_candidates(query)

        results: List[SimilarAlloy] = []
        for index in candidate_indices:
            row = self.df.iloc[index]
            properties = {
                prop: (None if pd.isna(row[prop]) else float(row[prop]))
                for prop in config.NUMERIC_PROPERTIES
            }
            match = evaluate_match(
                properties=properties,
                corrosion=row["corrosion_resistance"],
                application=row["application"],
                query=query,
            )
            results.append(SimilarAlloy(
                alloy_id=int(row["id"]),
                name=str(row["name"]),
                source=str(row["source"]),
                composition=row["composition"] or {},
                properties=properties,
                corrosion_resistance=row["corrosion_resistance"],
                application=row["application"] or None,
                match=match,
                distance=self._last_distances.get(index, 0.0),
                explanation=build_explanation(match),
            ))

        # Основной критерий — процент соответствия; при равенстве
        # предпочитаем запись, которая ближе по метрике kNN.
        results.sort(key=lambda item: (-item.percent, item.distance))
        return results[:top_n]

    # ------------------------------------------------------- вспомогательное

    def nearest_by_composition(self, composition_vector: np.ndarray) -> Optional[pd.Series]:
        """
        Находит запись БД с наиболее похожим химическим составом.
        Используется генеративной моделью для сравнения «новый сплав vs аналог».
        """
        if self.df is None or self.df.empty:
            return None
        matrix = element_matrix(self.df)
        distances = np.abs(matrix - composition_vector.reshape(1, -1)).sum(axis=1)
        return self.df.iloc[int(np.argmin(distances))]
