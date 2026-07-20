import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PKL = "movies_cleaned.pkl"
SIMILARITY_PKL = "similarity.pkl"
MOVIES_PKL = "movies.pkl"
MOVIE_INDICES_PKL = "movie_indices.pkl"

MAX_FEATURES = 5000
VOTE_COUNT_QUANTILE = 0.90

def load_movies(path: str) -> pd.DataFrame:
    """Load the cleaned movie dataframe from disk."""
    df = pd.read_pickle(path)
    print(f"Loaded movies: {df.shape}")
    return df

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Transform the "tags" column into a bag-of-words count matrix.

    Returns a sparse matrix of shape (n_movies, MAX_FEATURES).
    """
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
    # fit_transform learns the vocabulary from the tags and returns the counts.
    count_matrix = vectorizer.fit_transform(df["tags"])
    print(f"Feature matrix shape: {count_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return count_matrix

def compute_similarity(count_matrix: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine-similarity matrix and cast it to float32.

    float32 roughly halves the on-disk size versus the default float64 while
    keeping more than enough precision for similarity lookup.
    """
    similarity = cosine_similarity(count_matrix)
    similarity = similarity.astype(np.float32)
    print(f"Similarity matrix dtype: {similarity.dtype}")
    return similarity


def compute_weighted_rating(
    df: pd.DataFrame, quantile: float = VOTE_COUNT_QUANTILE
) -> pd.Series:

    # Global mean rating across the whole corpus.
    C = df["vote_average"].mean()
    # Minimum vote threshold (the chosen percentile of the vote-count distribution).
    m = df["vote_count"].quantile(quantile)

    R = df["vote_average"]
    v = df["vote_count"]

    weighted = (v / (v + m)) * R + (m / (v + m)) * C

    print(f"Global mean rating C = {C:.3f}")
    print(f"Vote-count threshold m (p{int(quantile * 100)}) = {m:.1f}")
    return weighted


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Scale a series linearly into the 0-1 range (min-max normalization)."""
    lo = series.min()
    hi = series.max()
    if hi == lo:
        # Avoid divide-by-zero for a constant series; map everything to 0.5.
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)

def build_title_index(df: pd.DataFrame) -> dict[str, int]:
    """Map each lowercase movie title to its row index for fast lookup."""
    # enumerate preserves row order, so the value is the position that matches
    # the corresponding row of the similarity matrix.
    return {title.lower(): i for i, title in enumerate(df["title"])}

def save_artifacts(
    similarity: np.ndarray, df: pd.DataFrame, title_index: dict[str, int]
) -> None:
    """Persist the similarity matrix, enriched dataframe, and title index."""
    with open(SIMILARITY_PKL, "wb") as f:
        pickle.dump(similarity, f)
    print(f"Saved similarity matrix to '{SIMILARITY_PKL}'")

    df.to_pickle(MOVIES_PKL)
    print(f"Saved movies (with weighted_rating) to '{MOVIES_PKL}'")

    with open(MOVIE_INDICES_PKL, "wb") as f:
        pickle.dump(title_index, f)
    print(f"Saved title index to '{MOVIE_INDICES_PKL}'")

def main() -> None:
    # Load data
    df = load_movies(INPUT_PKL)

    # Vectorize tags & compute similarity 
    count_matrix = build_feature_matrix(df)
    similarity = compute_similarity(count_matrix)

    # Weighted rating & normalization 
    weighted = compute_weighted_rating(df, VOTE_COUNT_QUANTILE)
    df["weighted_rating"] = min_max_normalize(weighted)

    # Title index 
    title_index = build_title_index(df)

    # Persist 
    save_artifacts(similarity, df, title_index)

    # Verification output 
    print("\n Similarity matrix shape")
    print(similarity.shape)

    print("\n Sample of 3 movies with weighted rating")
    sample = df[["title", "vote_average", "vote_count", "weighted_rating"]].head(3)
    with pd.option_context("display.max_colwidth", 60):
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
