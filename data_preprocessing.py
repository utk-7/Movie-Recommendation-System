"""
data_preprocessing.py
---------------------
Prepares the TMDB 5000 Movie Dataset for a hybrid recommendation system.

Pipeline:
  1. Load the two raw CSVs (movies + credits).
  2. Merge them on the movie id (and cross-check the title).
  3. Parse the JSON-like string columns with `ast.literal_eval`.
  4. Extract the top-3 cast members and the director.
  5. Build a cleaned, lowercased "tags" column (overview + genres +
     keywords + cast + director).
  6. Keep a tidy, analyst-friendly subset of columns.
  7. Handle missing values and persist the result as a pickle.

Everything runs offline from the CSV files in the project root.
"""

import ast
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOVIES_CSV = "tmdb_5000_movies.csv"
CREDITS_CSV = "tmdb_5000_credits.csv"
OUTPUT_PKL = "movies_cleaned.pkl"

# How many leading cast members to keep (ordered by billing "order").
TOP_CAST_COUNT = 3

# Columns we want to keep in the final, cleaned dataframe.
FINAL_COLUMNS = [
    "id",            # movie id
    "title",
    "overview",
    "genres",        # readable, comma-separated string
    "release_date",
    "vote_average",
    "vote_count",
    "tags",          # cleaned searchable text
]


# ---------------------------------------------------------------------------
# 1. Loading & merging
# ---------------------------------------------------------------------------
def load_and_merge(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load both CSVs and merge them on the movie id.

    The movies table keys on `id`; the credits table keys on `movie_id`.
    We merge on those keys and keep the movies `title` as the canonical one.
    """
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Both tables carry a `title`; drop the credits copy so the merge keeps a
    # single canonical `title` column (avoids title_x / title_y suffixes).
    credits = credits.drop(columns=["title"])

    # Merge credits onto movies via the shared movie identifier.
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="left")
    # `movie_id` is now redundant with `id`.
    df.drop(columns=["movie_id"], inplace=True)

    print(f"Loaded movies: {movies.shape}, credits: {credits.shape}")
    print(f"Merged dataframe shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. JSON parsing helpers
# ---------------------------------------------------------------------------
def parse_json_column(series: pd.Series) -> pd.Series:
    """Safely parse a column of JSON-like strings into Python objects.

    Returns an empty list for any value that cannot be parsed (NaN, malformed).
    """
    def _parse(value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
        return []

    return series.apply(_parse)


def extract_names(json_list, key: str = "name", top_n: int | None = None) -> list[str]:
    """Pull a list of `name` strings out of a parsed JSON list of dicts.

    `top_n` limits how many entries to keep (ordered as provided, e.g. by
    billing order for cast).
    """
    if not isinstance(json_list, list):
        return []
    names = [item.get(key, "") for item in json_list if isinstance(item, dict)]
    if top_n is not None:
        names = names[:top_n]
    return [n for n in names if n]  # drop empty/blank names


def extract_director(crew_list) -> str:
    """Find the director's name from a parsed crew list (job == 'Director')."""
    if not isinstance(crew_list, list):
        return ""
    for member in crew_list:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


# ---------------------------------------------------------------------------
# 3. Text cleaning helpers
# ---------------------------------------------------------------------------
def clean_token(text: str) -> str:
    """Clean a single token (name/keyword/genre) for use inside `tags`.

    Lowercases and strips ALL non-alphanumeric characters including internal
    spaces, so "Sam Worthington" -> "samworthington" and "culture clash"
    -> "cultureclash".
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep only letters/digits; this also removes spaces and punctuation.
    return re.sub(r"[^a-z0-9]", "", text)


def clean_overview(text: str) -> str:
    """Clean the free-text overview: lowercase, drop punctuation, collapse
    whitespace. Words are preserved (unlike `clean_token`) so the text stays
    readable/searchable.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace punctuation with a space, then collapse repeated spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 4. Feature construction
# ---------------------------------------------------------------------------
def build_genres_readable(genres_list) -> str:
    """Render genres as a readable, comma-separated string, e.g.
    'Action, Adventure, Fantasy'."""
    names = extract_names(genres_list)
    return ", ".join(names)


def build_tags(row) -> str:
    """Combine overview + genres + keywords + cast + director into one
    cleaned, lowercased string of tokens suitable for similarity search.
    """
    overview = clean_overview(row["overview"])
    genres = " ".join(clean_token(g) for g in extract_names(row["genres"]))
    keywords = " ".join(clean_token(k) for k in extract_names(row["keywords"]))
    cast = " ".join(clean_token(c) for c in row["cast_names"])
    director = clean_token(row["director"])

    # Join the parts and collapse any extra whitespace.
    tags = " ".join([overview, genres, keywords, cast, director])
    return re.sub(r"\s+", " ", tags).strip()


# ---------------------------------------------------------------------------
# 5. Main preprocessing routine
# ---------------------------------------------------------------------------
def preprocess() -> pd.DataFrame:
    # --- Load & merge -------------------------------------------------------
    df = load_and_merge(MOVIES_CSV, CREDITS_CSV)

    # --- Parse JSON-like columns -------------------------------------------
    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = parse_json_column(df[col])

    # --- Extract structured fields -----------------------------------------
    df["cast_names"] = df["cast"].apply(lambda x: extract_names(x, top_n=TOP_CAST_COUNT))
    df["director"] = df["crew"].apply(extract_director)

    # --- Build tags ---------------------------------------------------------
    df["tags"] = df.apply(build_tags, axis=1)

    # --- Build the readable genres string ----------------------------------
    df["genres"] = df["genres"].apply(build_genres_readable)

    # --- Clean the overview text -------------------------------------------
    df["overview"] = df["overview"].apply(clean_overview)

    # --- Handle missing values ---------------------------------------------
    # Drop rows that have no usable text content.
    df = df[df["overview"].str.len() > 0]
    df = df[df["tags"].str.len() > 0]

    # Fill remaining NaNs sensibly.
    # Numeric rating fields -> 0 when absent.
    df["vote_average"] = df["vote_average"].fillna(0)
    df["vote_count"] = df["vote_count"].fillna(0)
    # Date -> empty string when missing.
    df["release_date"] = df["release_date"].fillna("")
    # Genres/tags are already strings; guard against any stray NaN.
    df["genres"] = df["genres"].fillna("")
    df["tags"] = df["tags"].fillna("")

    # --- Keep only the requested columns -----------------------------------
    df = df[FINAL_COLUMNS].reset_index(drop=True)

    # --- Persist ------------------------------------------------------------
    df.to_pickle(OUTPUT_PKL)
    print(f"\nSaved cleaned data to '{OUTPUT_PKL}'")
    return df


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    movies_clean = preprocess()

    print("\n=== DataFrame shape ===")
    print(movies_clean.shape)

    print("\n=== First 3 rows ===")
    # Use to_string for a readable console dump; truncate long text columns.
    with pd.option_context("display.max_colwidth", 80):
        print(movies_clean.head(3).to_string())
