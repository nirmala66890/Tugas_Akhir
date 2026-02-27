from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD


@dataclass
class CollaborativeModel:
    model: SVD
    trainset: any


def train_svd_model(ratings_df: pd.DataFrame) -> CollaborativeModel:
    filtered = ratings_df[ratings_df["rating"] != -1].copy()
    reader = Reader(rating_scale=(1, 10))
    dataset = Dataset.load_from_df(filtered[["user_id", "anime_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    algo = SVD(n_factors=100, n_epochs=25, lr_all=0.005, reg_all=0.02, random_state=42)
    algo.fit(trainset)
    return CollaborativeModel(model=algo, trainset=trainset)


def get_user_cf_scores(
    collaborative_model: CollaborativeModel,
    user_id: int,
    candidate_anime_ids: Iterable[int],
) -> pd.DataFrame:
    rows: List[dict] = []
    for anime_id in candidate_anime_ids:
        pred = collaborative_model.model.predict(uid=user_id, iid=int(anime_id), verbose=False)
        rows.append({"anime_id": int(anime_id), "collaborative_score": float(pred.est)})
    return pd.DataFrame(rows)


def get_item_cf_scores(
    collaborative_model: CollaborativeModel,
    seed_anime_id: int,
    candidate_anime_ids: Iterable[int],
) -> pd.DataFrame:
    trainset = collaborative_model.trainset
    model = collaborative_model.model

    try:
        try:
            seed_inner = trainset.to_inner_iid(int(seed_anime_id))
        except ValueError:
            seed_inner = trainset.to_inner_iid(str(seed_anime_id))
    except ValueError:
        return pd.DataFrame(
            [{"anime_id": int(iid), "collaborative_score": 0.0} for iid in candidate_anime_ids]
        )

    seed_vec = model.qi[seed_inner]
    seed_norm = np.linalg.norm(seed_vec) + 1e-12

    rows: List[dict] = []
    for anime_id in candidate_anime_ids:
        try:
            try:
                cand_inner = trainset.to_inner_iid(int(anime_id))
            except ValueError:
                cand_inner = trainset.to_inner_iid(str(int(anime_id)))
            cand_vec = model.qi[cand_inner]
            score = float(np.dot(seed_vec, cand_vec) / (seed_norm * (np.linalg.norm(cand_vec) + 1e-12)))
        except ValueError:
            score = 0.0
        rows.append({"anime_id": int(anime_id), "collaborative_score": score})

    return pd.DataFrame(rows)
