import streamlit as st

from recommender import get_recommendations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOVIES_PKL = "movies.pkl"
COLS_PER_ROW = 4  # cards per row in the results grid (3-4 is comfortable)


# ---------------------------------------------------------------------------
# Data loading (cached; only reads the pickle once)
# ---------------------------------------------------------------------------
@st.cache_data
def load_titles() -> list[str]:
    """Return all movie titles, alphabetically sorted and de-duplicated."""
    import pandas as pd

    df = pd.read_pickle(MOVIES_PKL)
    titles = df["title"].dropna().astype(str).tolist()
    return sorted(set(titles))


def extract_year(release_date) -> str:
    """Pull the 4-digit year out of a 'YYYY-MM-DD' string, or '—' if absent."""
    if isinstance(release_date, str) and len(release_date) >= 4:
        year = release_date[:4]
        if year.isdigit():
            return year
    return "—"


st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Movie Recommender")
st.caption("Content-based suggestions blended with an IMDB-style weighted rating — fully offline, no external data fetched.")

with st.sidebar:
    st.header("How this works")
    st.write(
        "Pick a movie and we find the most similar titles in the dataset."
    )
    st.markdown(
        "**1. Content similarity** — each movie is described by a bag-of-words "
        "of its tags (overview, genres, keywords, cast, director) and compared "
        "with cosine similarity."
    )
    st.markdown(
        "**2. Weighted rating** — a Bayesian (IMDB-style) score so that movies "
        "with few votes aren't unfairly boosted."
    )
    st.markdown(
        "**3. Blend** — `blended = 0.7 × similarity + 0.3 × weighted rating`, "
        "then the top matches are shown."
    )
    st.info("All data is loaded from local pickle files. No network calls are made.")

if "results" not in st.session_state:
    st.session_state.results = None
if "movie_select" not in st.session_state:
    st.session_state.movie_select = load_titles()[0]


titles = load_titles()

selected = st.selectbox(
    "Search a movie",
    titles,
    index=titles.index(st.session_state.movie_select)
    if st.session_state.movie_select in titles
    else 0,
    key="movie_select",
)

if st.button("Recommend", type="primary"):
    st.session_state.results = get_recommendations(selected)

results = st.session_state.results

if results is None:
    st.write("Select a movie above and hit **Recommend** to see suggestions.")

elif isinstance(results, dict):
    # The recommender signals "not found" with an error dict + suggestions.
    st.error(results.get("error", "Movie not found."))
    suggestions = results.get("suggestions", [])
    if suggestions:
        st.write("Did you mean:")
        for suggestion in suggestions:
            if st.button(suggestion, key=f"sug_{suggestion}"):
                # Re-run the recommendation for the suggested title.
                st.session_state.movie_select = suggestion
                st.session_state.results = get_recommendations(suggestion)
                st.rerun()
    else:
        st.write("No close matches found.")

elif len(results) == 0:
    st.warning("No recommendations found for that movie.")

else:
    st.success(f"Top {len(results)} recommendations:")
    for row_start in range(0, len(results), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for col, rec in zip(cols, results[row_start : row_start + COLS_PER_ROW]):
            with col:
                st.markdown(f"### {rec['title']}")
                st.write(rec["genres"])
                year = extract_year(rec["release_date"])
                st.write(f"📅 {year}")
                st.write(f"⭐ {rec['vote_average']:.1f}")
                st.write(f"Blended match: **{rec['blended_score']:.3f}**")
                st.divider()
