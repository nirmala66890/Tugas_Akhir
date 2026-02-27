from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

MODELS_DIR = Path("app/models")
FRONTEND_DIR = Path("frontend")


def load_resources():
    required = ["svd_model.pkl", "tfidf.pkl", "similarity.pkl", "anime_df.pkl", "ratings_df.pkl"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            "Missing trained model artifacts: "
            + ", ".join(missing)
            + ". Run `python train.py` first."
        )

    return {
        "svd_model": joblib.load(MODELS_DIR / "svd_model.pkl"),
        "content_model": {
            "tfidf": joblib.load(MODELS_DIR / "tfidf.pkl"),
            "similarity_matrix": joblib.load(MODELS_DIR / "similarity.pkl"),
        },
        "anime_df": joblib.load(MODELS_DIR / "anime_df.pkl"),
        "ratings_df": joblib.load(MODELS_DIR / "ratings_df.pkl"),
    }


def _hydrate_content_model(resources: dict):
    from app.recommender.content_based import ContentModel

    resources["content_model"] = ContentModel(
        tfidf=resources["content_model"]["tfidf"],
        tfidf_matrix=None,
        similarity_matrix=resources["content_model"]["similarity_matrix"],
    )
    return resources


app = FastAPI(title="Hybrid Anime Recommender", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    app.state.resources = _hydrate_content_model(load_resources())


app.include_router(router)
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")
