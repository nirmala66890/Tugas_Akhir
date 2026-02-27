from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .collaborative import CollaborativeModel, get_item_cf_scores, get_user_cf_scores
from .content_based import ContentModel


def _normalize_scores(df: pd.DataFrame, column: str, out_col: str) -> pd.DataFrame:
    if df.empty:
        df[out_col] = []
        return df

    scaler = MinMaxScaler()
    values = df[[column]].values
    if np.allclose(values.max(), values.min()):
        df[out_col] = 0.5
    else:
        df[out_col] = scaler.fit_transform(values)
    return df


def get_hybrid_recommendations_for_user(
    user_id: int,
    anime_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    collaborative_model: CollaborativeModel,
    content_model: ContentModel,
    alpha: float = 0.6,
    top_n: int = 10,
) -> List[dict]:
    watched_ids = set(ratings_df[ratings_df["user_id"] == user_id]["anime_id"].tolist())
    candidates = anime_df[~anime_df["anime_id"].isin(watched_ids)].copy()
    if candidates.empty:
        return []

    cf_scores = get_user_cf_scores(collaborative_model, user_id, candidates["anime_id"].tolist())
    candidates = candidates.merge(cf_scores, on="anime_id", how="left")

    liked = ratings_df[(ratings_df["user_id"] == user_id) & (ratings_df["rating"] >= 7)]["anime_id"].tolist()
    data = anime_df.reset_index(drop=True)
    index_map = {int(aid): idx for idx, aid in enumerate(data["anime_id"].tolist())}

    content_scores = []
    for anime_id in candidates["anime_id"].tolist():
        idx = index_map.get(int(anime_id))
        if idx is None or not liked:
            content_scores.append(0.0)
            continue
        sims = []
        for liked_id in liked:
            liked_idx = index_map.get(int(liked_id))
            if liked_idx is not None:
                sims.append(float(content_model.similarity_matrix[idx, liked_idx]))
        content_scores.append(float(np.mean(sims)) if sims else 0.0)

    candidates["content_score"] = content_scores

    candidates = _normalize_scores(candidates, "collaborative_score", "cf_norm")
    candidates = _normalize_scores(candidates, "content_score", "content_norm")
    candidates["score"] = alpha * candidates["cf_norm"] + (1 - alpha) * candidates["content_norm"]

    top = candidates.sort_values("score", ascending=False).head(top_n)
    return [
        {"anime_id": int(row.anime_id), "title": str(row.title), "score": float(row.score)}
        for _, row in top.iterrows()
    ]


def get_hybrid_recommendations_for_anime(
    anime_id: int,
    anime_df: pd.DataFrame,
    collaborative_model: CollaborativeModel,
    content_model: ContentModel,
    alpha: float = 0.6,
    top_n: int = 10,
) -> List[dict]:
    data = anime_df.reset_index(drop=True).copy()
    if anime_id not in data["anime_id"].values:
        return []

    candidates = data[data["anime_id"] != anime_id].copy()
    seed_idx = int(data.index[data["anime_id"] == anime_id][0])

    # Content similarity against the selected anime.
    content_scores = content_model.similarity_matrix[seed_idx]
    candidates["content_score"] = candidates.index.map(lambda idx: float(content_scores[int(idx)]))

    # Collaborative item-item similarity from learned SVD item factors.
    cf_scores = get_item_cf_scores(collaborative_model, anime_id, candidates["anime_id"].tolist())
    candidates = candidates.merge(cf_scores, on="anime_id", how="left")

    candidates = _normalize_scores(candidates, "collaborative_score", "cf_norm")
    candidates = _normalize_scores(candidates, "content_score", "content_norm")
    candidates["score"] = alpha * candidates["cf_norm"] + (1 - alpha) * candidates["content_norm"]

    top = candidates.sort_values("score", ascending=False).head(top_n)
    return [
        {"anime_id": int(row.anime_id), "title": str(row.title), "score": float(row.score)}
        for _, row in top.iterrows()
    ]
