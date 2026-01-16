from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator


DATA_PATH = Path(__file__).resolve().parent.parent / "videos_data.csv"
ALGORITHM_VERSION = "1.0"
TOP_K = 5


class RecommendRequest(BaseModel):
    """Запрос на получение рекомендаций видео"""
    user_id: str = Field(..., description="Уникальный идентификатор пользователя (id)")
    watched_videos: List[str] = Field(
        default_factory=list, description="Список id видео, уже просмотренных пользователем"
    )
    liked_categories: List[str] = Field(
        default_factory=list, description="Категории видео, которые нравятся пользователю"
    )

    @validator("watched_videos", "liked_categories", each_item=True)
    def strip_items(cls, value: str) -> str:
        return value.strip()


class RecommendResponse(BaseModel):
    """Ответ с рекомендациями видео"""
    user_id: str = Field(..., description="id пользователя из запроса")
    recommendations: List[str] = Field(..., description="Список рекомендованных id видео")
    algorithm_version: str = Field(..., description="Версия сервиса")


def _parse_embedding(raw: str) -> np.ndarray:
    """Преобразует эмбеддинг в numpy массив."""
    try:
        
        cleaned = raw.strip().strip('"').strip("'")
        arr = ast.literal_eval(cleaned)
        return np.asarray(arr, dtype=float)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Ошибка парсинга: {raw}") from exc


def _load_video_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {path}")

    df = pd.read_csv(path, sep=";")
    required_columns = {"video_id", "title", "category", "embedding"}
    if not required_columns.issubset(set(df.columns)):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Отсутствуют обязательные колонки в файле данных: {missing}")

    df["embedding"] = df["embedding"].apply(_parse_embedding)
    return df


def _build_category_centroids(df: pd.DataFrame) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for category, group in df.groupby("category"):
        vectors = np.stack(group["embedding"].to_numpy())
        centroids[category] = vectors.mean(axis=0)
    return centroids


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _score_videos(
    df: pd.DataFrame, centroids: dict[str, np.ndarray], liked_categories: List[str]
) -> pd.Series:
    """Оценивает видео по схожести с центроидами предпочитаемых категорий."""
    if liked_categories:
        active_centroids = [
            centroids[c] for c in liked_categories if c in centroids
        ]
        if not active_centroids:
            
            active_centroids = [np.stack(list(centroids.values())).mean(axis=0)]
    else:
        active_centroids = [np.stack(list(centroids.values())).mean(axis=0)]

    def _score(row: pd.Series) -> float:
        embedding = row["embedding"]
        return max(_cosine_similarity(embedding, centroid) for centroid in active_centroids)

    return df.apply(_score, axis=1)


def recommend_videos(
    df: pd.DataFrame, centroids: dict[str, np.ndarray], request: RecommendRequest
) -> List[str]:
    watched = set(request.watched_videos)
    liked_categories = [c for c in request.liked_categories if c]

    candidates = df[~df["video_id"].isin(watched)].copy()
    if liked_categories:
        candidates = candidates[candidates["category"].isin(liked_categories)]
        
        if candidates.empty:
            candidates = df[~df["video_id"].isin(watched)].copy()

    candidates["score"] = _score_videos(candidates, centroids, liked_categories)
    ranked = candidates.sort_values(by="score", ascending=False)
    return ranked["video_id"].head(TOP_K).tolist()


videos_df = _load_video_data(DATA_PATH)
category_centroids = _build_category_centroids(videos_df)

app = FastAPI(
    title="Сервис рекомендаций видео",
    description="Рекомендательная система коротких видео на основе ML-алгоритма",
    version=ALGORITHM_VERSION
)


@app.get("/health", summary="Проверка работоспособности сервиса")
def health() -> dict[str, str]:
    """Возвращает статус работы сервиса"""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse, summary="Получить рекомендации видео")
def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Возвращает персонализированные рекомендации видео на основе:
    - Просмотренных пользователем видео (исключаются из рекомендаций)
    - Категорий, которые нравятся пользователю
    
    Алгоритм использует косинусное сходство эмбеддингов для ранжирования.
    """
    try:
        recommendations = recommend_videos(videos_df, category_centroids, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        algorithm_version=ALGORITHM_VERSION,
    )
