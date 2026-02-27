from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentModel:
    tfidf: TfidfVectorizer
    tfidf_matrix: np.ndarray
    similarity_matrix: np.ndarray


def train_content_model(anime_df: pd.DataFrame) -> ContentModel:
    data = anime_df.reset_index(drop=True).copy()
    data["genre"] = data["genre"].fillna("")
    data["synopsis"] = data["synopsis"].fillna("")
    data["combined_text"] = data["genre"] + " " + data["synopsis"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = tfidf.fit_transform(data["combined_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return ContentModel(
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        similarity_matrix=similarity_matrix,
    )


def get_similar_by_title(content_model: ContentModel, anime_df: pd.DataFrame, anime_title: str, top_n: int = 10):
    data = anime_df.reset_index(drop=True)
    lower_titles = data["title"].astype(str).str.lower()
    exact = data[lower_titles == anime_title.lower()]

    if exact.empty:
        partial = data[lower_titles.str.contains(anime_title.lower(), regex=False)]
        if partial.empty:
            return []
        target_idx = int(partial.index[0])
    else:
        target_idx = int(exact.index[0])

    sims = content_model.similarity_matrix[target_idx]
    ranked_indices = np.argsort(sims)[::-1]

    recs = []
    for idx in ranked_indices:
        if idx == target_idx:
            continue
        anime_id = int(data.iloc[idx]["anime_id"])
        title = str(data.iloc[idx]["title"])
        recs.append({"anime_id": anime_id, "title": title, "score": float(sims[idx])})
        if len(recs) >= top_n:
            break
    return recs
