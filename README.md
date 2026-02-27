# Hybrid Anime Recommender (Catalog-Driven)

Production-ready anime recommendation system using a **Hybrid Filtering** approach:

- Collaborative Filtering: **SVD** (`surprise`)
- Content-Based Filtering: **TF-IDF** on **genre + synopsis** with cosine similarity
- Hybrid scoring:

```text
final_score = alpha * collaborative_score + (1 - alpha) * content_score
```

Default `alpha = 0.6`, with normalization before blending.

## Features

- Automatic dataset provisioning to `data/`:
  - Tries Kaggle dataset `CooperUnion/anime-recommendations-database`
  - Renames `rating.csv` to `ratings.csv`
  - Fallback to public mirrors when Kaggle is unavailable
- Catalog-style web UI:
  - Homepage displays anime cards (title, genre, synopsis preview)
  - Click card to open detail section and show hybrid recommendations
- FastAPI endpoints for catalog, detail, and recommendation flows.

## API

- `GET /health`
- `GET /anime?limit=80&q=naruto`
- `GET /anime/{anime_id}`
- `GET /recommend/anime/{anime_id}`
- `GET /recommend/user/{user_id}` (optional backend endpoint retained)

## Train

```bash
pip install -r requirements.txt
python train.py
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:
- `http://localhost:8000/`

## Docker

```bash
docker build -t anime-recommender .
docker run --rm -p 8000:8000 anime-recommender
```
