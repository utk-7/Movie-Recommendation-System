"""
recommender.py
--------------
Hybrid movie recommender that blends content-based similarity with an
IMDB-style weighted quality rating.

Given a movie title it returns the `top_n` most similar movies, ranked by a
blended score:

    blended_score = alpha * content_similarity + beta * weighted_rating

The artifacts consumed here (`movies.pkl`, `similarity.pkl`, `movie_indices.pkl`)
are produced by `vectorize_and_similarity.py`. Everything runs offline.

On a successful lookup the function returns a *list* of dicts. If the requested
title is not found it returns an *error dict* (instead of raising) that also
lists a few close-matching titles, so callers can surface a helpful message
without crashing.
"""

import difflib
import pickle

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOVIES_PKL = "movies.pkl"
SIMILARITY_PKL = "similarity.pkl"
MOVIE_INDICES_PKL = "movie_indices.pkl"

# How many close-match suggestions to offer when a title is not found.
MAX_SUGGESTIONS = 3
# Minimum fuzzy-match score (0-1) required to suggest a title.
SUGGESTION_CUTOFF = 0.6

# Columns copied verbatim from the dataframe into each recommendation.
RESULT_COLUMNS = ["title", "genres", "vote_average", "release_date"]

# Module-level cache so the (large) artifacts are loaded only once per process.
_movies_df = None
_similarity = None
_title_index = None


# ---------------------------------------------------------------------------
# Artifact loading (lazily cached)
# ---------------------------------------------------------------------------
def _load_artifacts() -> tuple[pd.DataFrame, object, dict]:
    """Load and cache the three pickled artifacts.

    Returns `(movies_df, similarity_matrix, title_index)`. The first call
    reads the files from disk; subsequent calls return the cached objects.
    """
    global _movies_df, _similarity, _title_index

    if _movies_df is None or _similarity is None or _title_index is None:
        _movies_df = pd.read_pickle(MOVIES_PKL)
        with open(SIMILARITY_PKL, "rb") as f:
            _similarity = pickle.load(f)
        with open(MOVIE_INDICES_PKL, "rb") as f:
            _title_index = pickle.load(f)

    return _movies_df, _similarity, _title_index


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def _suggest_titles(missing_title: str, title_index: dict, movies_df: pd.DataFrame,
                    max_suggestions: int = MAX_SUGGESTIONS) -> list[str]:
    """Return up to `max_suggestions` close-matching original-cased titles.

    Uses `difflib.get_close_matches` against the lowercase keys of the title
    index, then maps each match back to its original (properly cased) title.
    """
    matches = difflib.get_close_matches(
        missing_title.lower().strip(),
        list(title_index.keys()),
        n=max_suggestions,
        cutoff=SUGGESTION_CUTOFF,
    )
    # Map the matched lowercase keys to real titles via the dataframe.
    suggestions = []
    for key in matches:
        row_idx = title_index[key]
        original = movies_df.iloc[row_idx]["title"]
        suggestions.append(str(original))
    return suggestions


# ---------------------------------------------------------------------------
# Core recommendation
# ---------------------------------------------------------------------------
def get_recommendations(movie_title: str, top_n: int = 8,
                        alpha: float = 0.7, beta: float = 0.3):
    """Recommend movies similar to `movie_title`.

    Parameters
    ----------
    movie_title : str
        Title to find recommendations for (matched case-insensitively, trimmed).
    top_n : int
        Number of recommendations to return (default 8).
    alpha : float
        Weight for the content similarity component (default 0.7).
    beta : float
        Weight for the normalized weighted-rating component (default 0.3).

    Returns
    -------
    list[dict]  on success: each dict has `title`, `genres`, `vote_average`,
                `release_date`, and `blended_score` (rounded to 3 decimals).
    dict        on failure: {"error": <message>, "suggestions": [<titles>]}.
    """
    movies_df, similarity, title_index = _load_artifacts()

    # --- Resolve the requested title ---------------------------------------
    key = movie_title.lower().strip()
    if key not in title_index:
        suggestions = _suggest_titles(key, title_index, movies_df)
        return {
            "error": (
                f"Movie '{movie_title}' not found in the dataset."
            ),
            "suggestions": suggestions,
        }

    movie_idx = title_index[key]
    # Content similarity scores of this movie against every other movie.
    content_scores = similarity[movie_idx]

    recommendations = []
    for other_idx in range(len(movies_df)):
        if other_idx == movie_idx:
            continue  # never recommend the input movie itself

        content_sim = float(content_scores[other_idx])
        weighted = float(movies_df.iloc[other_idx]["weighted_rating"])
        blended = alpha * content_sim + beta * weighted

        row = movies_df.iloc[other_idx]
        recommendations.append(
            {
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "vote_average": float(row["vote_average"]),
                "release_date": str(row["release_date"]),
                "blended_score": round(blended, 3),
            }
        )

    # Highest blended score first, then trim to the requested count.
    recommendations.sort(key=lambda r: r["blended_score"], reverse=True)
    return recommendations[:top_n]


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------
def _print_table(recs) -> None:
    """Pretty-print the result: a table on success, a message on failure."""
    if isinstance(recs, dict):
        print(recs["error"])
        if recs["suggestions"]:
            print("Did you mean:")
            for title in recs["suggestions"]:
                print(f"  - {title}")
        else:
            print("No close matches found.")
        return

    # Compute column widths from the data for a tidy aligned table.
    headers = ["#", "Title", "Genres", "Vote", "Release", "Blended"]
    rows = []
    for i, r in enumerate(recs, start=1):
        rows.append([
            str(i),
            r["title"],
            r["genres"],
            f"{r['vote_average']:.1f}",
            r["release_date"],
            f"{r['blended_score']:.3f}",
        ])

    widths = [
        max(len(headers[c]), *(len(row[c]) for row in rows))
        for c in range(len(headers))
    ]
    sep = "-+-".join("-" * w for w in widths)
    line_fmt = " | ".join(f"{{:<{w}}}" for w in widths)

    print(line_fmt.format(*headers))
    print(sep)
    for row in rows:
        print(line_fmt.format(*row))


if __name__ == "__main__":
    results = get_recommendations("Avatar")
    print("=== Recommendations for 'Avatar' ===")
    _print_table(results)
