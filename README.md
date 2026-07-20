# 🎬 Movie Recommendation System

A **hybrid content-based movie recommender** built entirely from the
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
Given a movie, it returns the most similar titles, ranked by a blend of
**content similarity** and an **IMDB-style weighted rating**.

There are **no external API calls** — no live TMDB requests, no API keys, no
secrets. Once the local data files are in place, everything runs offline from
pickled artifacts produced by the pipeline.

## Project layout

| File | Purpose |
|------|---------|
| `data_preprocessing.py` | Loads + merges the two raw CSVs, parses JSON columns, builds a cleaned `tags` string, and writes `movies_cleaned.pkl`. |
| `vectorize_and_similarity.py` | Vectorizes `tags`, computes the cosine-similarity matrix, computes a normalized weighted rating, and writes `similarity.pkl`, `movies.pkl`, `movie_indices.pkl`. |
| `recommender.py` | Core logic: `get_recommendations(title, top_n, alpha, beta)`. Standalone-testable. |
| `app.py` | Streamlit frontend (searchable selectbox + recommendation grid). |
| `requirements.txt` | Minimal Python dependencies. |

## Setup

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd Movie-Recommendation-System
   ```

2. **Add the raw data files** to the project root:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`

   (These come from the TMDB 5000 Kaggle dataset. They are not committed because
   of their size.)

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Run the pipeline (in order)

The pipeline builds the artifacts the recommender and UI consume. Run each step
once (or whenever you change the raw data):

```bash
# 1. Clean + preprocess the raw CSVs -> movies_cleaned.pkl
python data_preprocessing.py

# 2. Vectorize, compute similarity + weighted rating -> similarity.pkl,
#    movies.pkl, movie_indices.pkl
python vectorize_and_similarity.py

# 3. Launch the Streamlit frontend
streamlit run app.py
```

You can also test the recommender without the UI:

```bash
python recommender.py   # prints an Avatar recommendation table
```

## How the recommendation works

Each movie is described by a **`tags`** string built from its overview, genres,
keywords, top cast, and director. The pipeline:

1. **Vectorizes** `tags` with a `CountVectorizer` (top 5000 features, English
   stop words removed) into a bag-of-words matrix.
2. **Computes content similarity** as the pairwise cosine similarity between
   movies — `similarity[i][j]` is how content-similar movie *j* is to movie *i*.
3. **Computes a weighted rating** (Bayesian / IMDB-style) so movies with few
   votes aren't unfairly boosted by a single high score:

   ```
   WR = (v / (v + m)) * R + (m / (v + m)) * C
   ```
   where `R` = movie vote average, `v` = vote count, `C` = global mean vote
   average, and `m` = the 90th-percentile vote count across the corpus. This is
   then min–max normalized to the 0–1 range and stored as `weighted_rating`.

4. **Blends** the two signals. For every candidate movie *j* (excluding the
   input movie itself):

   ```
   blended_score = alpha * content_similarity + beta * weighted_rating
   ```

   with defaults **`alpha = 0.7`** and **`beta = 0.3`** (tunable via
   `get_recommendations(...)` or a future UI control). Results are sorted by
   `blended_score` descending and the top `top_n` are returned.

If the requested title isn't found, the recommender returns close-match
suggestions (via `difflib`) that the UI surfaces as one-click buttons.

## Deploy on Streamlit Community Cloud

No secrets or API keys are required — the app is fully self-contained.

1. Push this repo to **GitHub** (make sure `tmdb_5000_movies.csv` and
   `tmdb_5000_credits.csv` are present, or have the pipeline artifacts already
   committed; the deployed app needs the `.pkl` files it loads at runtime).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with
   GitHub.
3. Click **"New app"**, select your repository, branch, and set the main file
   path to `app.py`.
4. Leave the **Secrets** section empty — there is nothing to configure.
5. Click **Deploy**. Streamlit will `pip install -r requirements.txt` and launch
   `app.py`.

> **Tip:** Streamlit Community Cloud starts from a clean environment and does not
> run `data_preprocessing.py` / `vectorize_and_similarity.py` for you. Either
> commit the generated `*.pkl` files, or add a small bootstrap step that runs the
> pipeline on first launch if you want the app to regenerate them.
