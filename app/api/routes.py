from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.recommender.hybrid import (
    get_hybrid_recommendations_for_anime,
    get_hybrid_recommendations_for_user,
)

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "OK"}


@router.get("/anime")
def list_anime(request: Request, limit: int = Query(default=60, ge=1, le=500), q: str | None = None):
    anime_df = request.app.state.resources["anime_df"].copy()
    if q:
        anime_df = anime_df[anime_df["title"].str.contains(q, case=False, na=False)]

    anime_df = anime_df.head(limit)
    return {
        "anime": [
            {
                "anime_id": int(row.anime_id),
                "title": str(row.title),
                "genre": str(row.genre),
                "synopsis": str(row.synopsis),
            }
            for _, row in anime_df.iterrows()
        ]
    }


@router.get("/anime/{anime_id}")
def anime_detail(anime_id: int, request: Request):
    anime_df = request.app.state.resources["anime_df"]
    match = anime_df[anime_df["anime_id"] == anime_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Anime not found")

    row = match.iloc[0]
    return {
        "anime_id": int(row.anime_id),
        "title": str(row.title),
        "genre": str(row.genre),
        "synopsis": str(row.synopsis),
    }


@router.get("/recommend/user/{user_id}")
def recommend_by_user(user_id: int, request: Request):
    resources = request.app.state.resources
    recs = get_hybrid_recommendations_for_user(
        user_id=user_id,
        anime_df=resources["anime_df"],
        ratings_df=resources["ratings_df"],
        collaborative_model=resources["svd_model"],
        content_model=resources["content_model"],
        alpha=0.6,
        top_n=10,
    )
    return {"recommendations": recs}


@router.get("/recommend/anime/{anime_id}")
def recommend_by_anime_id(anime_id: int, request: Request):
    resources = request.app.state.resources
    recs = get_hybrid_recommendations_for_anime(
        anime_id=anime_id,
        anime_df=resources["anime_df"],
        collaborative_model=resources["svd_model"],
        content_model=resources["content_model"],
        alpha=0.6,
        top_n=10,
    )
    if not recs:
        raise HTTPException(status_code=404, detail="Anime not found or no recommendations available")

    anime_df = resources["anime_df"]
    anime_map = {int(row.anime_id): row for _, row in anime_df.iterrows()}
    enriched = []
    for item in recs:
        row = anime_map.get(int(item["anime_id"]))
        enriched.append({
            "anime_id": int(item["anime_id"]),
            "title": str(item["title"]),
            "genre": str(row.genre) if row is not None else "",
            "synopsis": str(row.synopsis) if row is not None else "",
            "score": float(item["score"]),
        })
    return {"recommendations": enriched}
