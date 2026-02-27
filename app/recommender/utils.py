from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

DATA_DIR = Path("data")
ANIME_PATH = DATA_DIR / "anime.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"

# Public fallback mirrors for the CooperUnion dataset.
FALLBACK_ANIME_URLS = [
    "https://raw.githubusercontent.com/ashimanur/FastAPI-Anime-Recommender/main/data/anime.csv",
    "https://raw.githubusercontent.com/mhsultan21/anime-recommender-system/main/anime.csv",
]

FALLBACK_RATING_URLS = [
    "https://raw.githubusercontent.com/ashimanur/FastAPI-Anime-Recommender/main/data/rating.csv",
    "https://raw.githubusercontent.com/mhsultan21/anime-recommender-system/main/rating.csv",
]


def _download_from_url(url: str, destination: Path) -> bool:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return True
    except Exception:
        return False


def _try_kaggle_download() -> bool:
    """Attempt Kaggle API download when credentials are available."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return False

    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and not (
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    ):
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "CooperUnion/anime-recommendations-database",
            path=str(DATA_DIR),
            unzip=True,
            quiet=False,
        )
        return True
    except Exception:
        return False


def _ensure_required_columns(anime_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    anime_df = anime_df.copy()
    ratings_df = ratings_df.copy()

    rename_map = {"name": "title", "rating": "avg_rating"}
    anime_df = anime_df.rename(columns=rename_map)

    for col in ["anime_id", "title", "genre"]:
        if col not in anime_df.columns:
            raise ValueError(f"anime.csv missing required column: {col}")

    if "synopsis" not in anime_df.columns:
        anime_df["synopsis"] = ""

    for col in ["anime_id", "title", "genre", "synopsis"]:
        anime_df[col] = anime_df[col].fillna("")

    ratings_df = ratings_df.rename(columns={"rating": "rating"})
    for col in ["user_id", "anime_id", "rating"]:
        if col not in ratings_df.columns:
            raise ValueError(f"ratings.csv missing required column: {col}")

    ratings_df = ratings_df[["user_id", "anime_id", "rating"]].copy()
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")
    ratings_df = ratings_df.dropna(subset=["user_id", "anime_id", "rating"])

    return anime_df[["anime_id", "title", "genre", "synopsis"]], ratings_df


def ensure_dataset() -> Tuple[Path, Path]:
    """Ensure anime.csv and ratings.csv exist in data/ with required schema."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ANIME_PATH.exists() and RATINGS_PATH.exists():
        return ANIME_PATH, RATINGS_PATH

    # Step 1: try Kaggle.
    _try_kaggle_download()

    # Handle original naming from source dataset.
    rating_csv = DATA_DIR / "rating.csv"
    if rating_csv.exists() and not RATINGS_PATH.exists():
        rating_csv.rename(RATINGS_PATH)

    # Step 2: fallback mirrors.
    if not ANIME_PATH.exists():
        for url in FALLBACK_ANIME_URLS:
            if _download_from_url(url, ANIME_PATH):
                break

    if not RATINGS_PATH.exists():
        for url in FALLBACK_RATING_URLS:
            if _download_from_url(url, RATINGS_PATH):
                break

    if not ANIME_PATH.exists() or not RATINGS_PATH.exists():
        raise FileNotFoundError(
            "Unable to download dataset from Kaggle or fallback sources. "
            "Provide data/anime.csv and data/ratings.csv manually."
        )

    # Validate and normalize schema.
    anime_df = pd.read_csv(ANIME_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)
    anime_df, ratings_df = _ensure_required_columns(anime_df, ratings_df)
    anime_df.to_csv(ANIME_PATH, index=False)
    ratings_df.to_csv(RATINGS_PATH, index=False)

    return ANIME_PATH, RATINGS_PATH


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    anime_path, ratings_path = ensure_dataset()
    anime_df = pd.read_csv(anime_path)
    ratings_df = pd.read_csv(ratings_path)

    anime_df["anime_id"] = pd.to_numeric(anime_df["anime_id"], errors="coerce")
    ratings_df["anime_id"] = pd.to_numeric(ratings_df["anime_id"], errors="coerce")
    ratings_df["user_id"] = pd.to_numeric(ratings_df["user_id"], errors="coerce")
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")

    anime_df = anime_df.dropna(subset=["anime_id", "title"]).copy()
    ratings_df = ratings_df.dropna(subset=["user_id", "anime_id", "rating"]).copy()

    anime_df["anime_id"] = anime_df["anime_id"].astype(int)
    ratings_df["anime_id"] = ratings_df["anime_id"].astype(int)
    ratings_df["user_id"] = ratings_df["user_id"].astype(int)

    return anime_df, ratings_df
