import numpy as np
import pandas as pd
import streamlit as st

from helpers import (MODEL_NAMES, RAW_COLS, build_features, load_metrics,
                     load_models, predict, predict_all)

st.set_page_config(page_title="Book Rating Predictor", page_icon="📚", layout="wide")


@st.cache_resource
def get_models():
    return load_models()


models = get_models()
metrics = load_metrics()
languages = sorted(models["pre"].named_transformers_["cat_low"].categories_[0])
DEFAULT_LANG = "eng" if "eng" in languages else languages[0]

# ---------------- Example presets ----------------
PRESETS = {
    "Custom (enter your own)": None,
    "Hugely popular fantasy": dict(ratings=2_000_000, work_ratings=2_200_000, reviews=50_000, to_read=100_000,
                                   reading=30_000, editions=300, year=2003, language="eng",
                                   authors="J.K. Rowling", tags="fantasy, young-adult, fiction, magic"),
    "Classic novel": dict(ratings=900_000, work_ratings=1_000_000, reviews=30_000, to_read=60_000,
                          reading=10_000, editions=900, year=1925, language="eng",
                          authors="F. Scott Fitzgerald", tags="classics, fiction, literature, historical"),
    "Niche / lesser-known book": dict(ratings=3_000, work_ratings=3_500, reviews=300, to_read=800,
                                      reading=60, editions=5, year=2012, language="eng",
                                      authors="Unknown Author", tags="romance, contemporary"),
}
DEFAULTS = dict(ratings=50_000, work_ratings=60_000, reviews=3_000, to_read=5_000, reading=1_000,
                editions=50, year=2005, language=DEFAULT_LANG, authors="Stephen King",
                tags="horror, fiction, thriller")

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)


def apply_preset():
    p = PRESETS[st.session_state["preset"]]
    if p:
        for k, v in p.items():
            st.session_state[k] = v if k != "language" or v in languages else DEFAULT_LANG


def current_raw():
    s = st.session_state
    return pd.DataFrame([{
        "ratings_count": s["ratings"], "work_ratings_count": s["work_ratings"],
        "work_text_reviews_count": s["reviews"], "toread_count": s["to_read"],
        "currently_reading": s["reading"], "books_count": s["editions"],
        "original_publication_year": s["year"], "language_code": s["language"],
        "authors": s["authors"].strip(), "tag_name": s["tags"].strip(),
    }])


def rating_label(r):
    if r >= 4.25: return "🌟 Exceptional"
    if r >= 4.05: return "👍 Great"
    if r >= 3.85: return "🙂 Good / average"
    return "😐 Below average"


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", MODEL_NAMES, help="Random Forest scored best in the notebook.")
    m = metrics.get(model_name, {})
    if m:
        st.caption("Test-set performance of this model")
        c1, c2 = st.columns(2)
        c1.metric("MAE", m.get("Test MAE"))
        c2.metric("R²", m.get("Test R2 Score"))
    st.divider()
    st.caption("Dataset: Goodreads 10k books\n\nTarget: `average_rating`\n\n"
               "Predictions are estimates (R² ≈ 0.4–0.5) — not exact ratings.")

st.title("📚 Book Rating Predictor")
st.write("Predict a book's Goodreads **average rating** from its popularity, age, language, author and genre tags.")

tab_predict, tab_batch, tab_insights, tab_history = st.tabs(
    ["🔮 Predict", "📂 Batch (CSV)", "📊 Insights", "📜 History"])

# ---------------- TAB 1: Predict ----------------
with tab_predict:
    st.selectbox("Quick example", list(PRESETS), key="preset", on_change=apply_preset)

    left, right = st.columns(2)
    with left:
        st.number_input("Ratings count", min_value=0, step=1000, key="ratings")
        st.number_input("Work ratings count", min_value=0, step=1000, key="work_ratings")
        st.number_input("Text reviews count", min_value=0, step=100, key="reviews")
        st.number_input("To-read count", min_value=0, step=100, key="to_read")
        st.number_input("Currently reading", min_value=0, step=100, key="reading")
    with right:
        st.number_input("Editions (books count)", min_value=1, step=1, key="editions")
        st.slider("Publication year", min_value=1800, max_value=2026, key="year")
        st.selectbox("Language code", languages, key="language")
        st.text_input("Author(s)", key="authors")
        st.text_input("Tags / genres (comma separated)", key="tags")

    if st.button("Predict rating", type="primary"):
        raw = current_raw()
        preds = predict_all(models, build_features(raw))
        st.session_state["last"] = preds
        st.session_state["history"].append({
            "Author": st.session_state["authors"], "Year": st.session_state["year"],
            "Language": st.session_state["language"], "Tags": st.session_state["tags"],
            "Ratings": st.session_state["ratings"], "Model": model_name,
            "Predicted rating": round(preds[model_name], 2),
        })

    preds = st.session_state["last"]
    if preds:
        pred = preds[model_name]
        mae = metrics.get(model_name, {}).get("Test MAE", 0.15)
        st.divider()
        a, b, c = st.columns([1, 1, 2])
        a.metric(f"Predicted rating ({model_name})", f"{pred:.2f} / 5")
        b.metric("Likely range", f"{max(1, pred - mae):.2f} – {min(5, pred + mae):.2f}",
                 help="Prediction ± the model's average test error (MAE).")
        stars = int(round(pred))
        c.markdown(f"### {'★' * stars}{'☆' * (5 - stars)}\n{rating_label(pred)}")
        st.progress(pred / 5)

        st.subheader("All models side by side")
        comp = pd.DataFrame({"Model": list(preds), "Predicted rating": [round(v, 3) for v in preds.values()]})
        st.bar_chart(comp, x="Model", y="Predicted rating", horizontal=True)
        spread = max(preds.values()) - min(preds.values())
        st.write(f"**Average of all models:** {np.mean(list(preds.values())):.2f}  ·  "
                 f"**Disagreement (max − min):** {spread:.2f}")
        if spread > 0.2:
            st.info("The models disagree noticeably for this book, so treat the prediction with extra caution.")

# ---------------- TAB 2: Batch ----------------
with tab_batch:
    st.write("Upload a CSV to predict many books at once. Required columns:")
    st.code(", ".join(RAW_COLS))
    template = pd.DataFrame([{
        "ratings_count": 50000, "work_ratings_count": 60000, "work_text_reviews_count": 3000,
        "toread_count": 5000, "currently_reading": 1000, "books_count": 50,
        "original_publication_year": 2005, "language_code": "eng",
        "authors": "Stephen King", "tag_name": "horror, fiction"}])
    st.download_button("⬇️ Download CSV template", template.to_csv(index=False),
                       "template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            X = build_features(df[RAW_COLS])
            out = df.copy()
            out["predicted_rating"] = predict(models, model_name, X).round(3)
            st.success(f"Predicted {len(out)} books using {model_name}.")
            st.dataframe(out)
            counts, edges = np.histogram(out["predicted_rating"], bins=10)
            st.caption("Distribution of predicted ratings")
            st.bar_chart(pd.DataFrame({"Rating": [f"{e:.2f}" for e in edges[:-1]], "Books": counts}),
                         x="Rating", y="Books")
            st.download_button("⬇️ Download predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv")

# ---------------- TAB 3: Insights ----------------
with tab_insights:
    st.subheader("Model comparison (test set)")
    if metrics:
        mt = pd.DataFrame([{"Model": n, "MAE": v.get("Test MAE"), "RMSE": v.get("Test RMSE"),
                            "R²": v.get("Test R2 Score")} for n, v in metrics.items()])
        st.dataframe(mt, hide_index=True)
        c1, c2 = st.columns(2)
        c1.caption("Error (lower is better)")
        c1.bar_chart(mt, x="Model", y="MAE")
        c2.caption("R² (higher is better)")
        c2.bar_chart(mt, x="Model", y="R²")

    st.subheader("What drives the rating? (Random Forest feature importance)")
    rf = models["Random Forest"]
    names = [n.split("__", 1)[-1] for n in rf.named_steps["preprocessor"].get_feature_names_out()]
    imp = pd.Series(rf.named_steps["rf"].feature_importances_, index=names)
    # group one-hot language columns and tag components for readability
    grouped = imp.groupby(lambda n: "language_code" if n.startswith("language_code_")
                          else "tags (SVD)" if n.startswith("truncatedsvd") else n).sum()
    st.bar_chart(grouped.sort_values(ascending=False).head(12), horizontal=True)

    st.subheader("What-if explorer")
    st.caption(f"Uses the values currently entered in the Predict tab and the **{model_name}** model.")
    var = st.radio("Vary", ["Publication year", "Popularity (ratings, reviews, to-read together)"],
                   horizontal=True)
    base = current_raw()
    if var == "Publication year":
        grid = list(range(1900, 2027, 4))
        rows = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
        rows["original_publication_year"] = grid
        xs = grid
    else:
        grid = [0.1, 0.25, 0.5, 1, 2, 4, 10]
        rows = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
        for i, f in enumerate(grid):
            for c in ["ratings_count", "work_ratings_count", "work_text_reviews_count",
                      "toread_count", "currently_reading"]:
                rows.loc[i, c] = base.loc[0, c] * f
        xs = [f"{f}x" for f in grid]
    ys = predict(models, model_name, build_features(rows))
    st.line_chart(pd.DataFrame({"x": xs, "Predicted rating": ys}), x="x", y="Predicted rating")

# ---------------- TAB 4: History ----------------
with tab_history:
    hist = st.session_state["history"]
    if hist:
        hdf = pd.DataFrame(hist)
        st.dataframe(hdf, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download history", hdf.to_csv(index=False), "history.csv", "text/csv")
        if d2.button("🗑️ Clear history"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No predictions yet. Make one in the Predict tab.")