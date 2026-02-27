from __future__ import annotations

from pathlib import Path

import joblib

from app.recommender.collaborative import train_svd_model
from app.recommender.content_based import train_content_model
from app.recommender.utils import load_data

MODELS_DIR = Path("app/models")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    anime_df, ratings_df = load_data()

    svd_model = train_svd_model(ratings_df)
    content_model = train_content_model(anime_df)

    joblib.dump(svd_model, MODELS_DIR / "svd_model.pkl")
    joblib.dump(content_model.tfidf, MODELS_DIR / "tfidf.pkl")
    joblib.dump(content_model.similarity_matrix, MODELS_DIR / "similarity.pkl")
    joblib.dump(anime_df, MODELS_DIR / "anime_df.pkl")
    joblib.dump(ratings_df, MODELS_DIR / "ratings_df.pkl")

    print("Training completed. Models saved to app/models/")


if __name__ == "__main__":
    main()
