# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance

from textblob import TextBlob
import plotly.express as px
import pydeck as pdk


st.set_page_config(
    page_title="Financial Inclusion Tracker",
    layout="wide",
)


@st.cache_data
def load_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


def detect_default_target(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        name = col.lower()
        if "adoption" in name or "score" in name or "index" in name:
            return col
    return numeric_cols[0] if numeric_cols else None


def overview_page(df):
    st.subheader("Dataset Overview")

    st.write(f"**Rows:** {df.shape[0]:,}  |  **Columns:** {df.shape[1]:,}")
    st.dataframe(df.head())

    with st.expander("Column information"):
        info_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "non_nulls": df.notnull().sum().values,
                "nulls": df.isnull().sum().values,
            }
        )
        st.dataframe(info_df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        with st.expander("Summary statistics (numeric columns)"):
            st.dataframe(df[numeric_cols].describe().T)

        with st.expander("Quick distribution view"):
            col = st.selectbox("Choose a numeric column", numeric_cols, key="overview_dist")
            fig = px.histogram(df, x=col, nbins=30)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected for summary statistics.")


def ml_page(df):
    st.subheader("ML Model – Predict Adoption Score Across Districts")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available. ML model cannot be built.")
        return

    default_target = detect_default_target(df)
    target_col = st.selectbox(
        "Select target column (Adoption score / Inclusion index)",
        numeric_cols,
        index=numeric_cols.index(default_target) if default_target in numeric_cols else 0,
    )

    candidate_features = [c for c in df.columns if c != target_col]
    feature_cols = st.multiselect(
        "Select feature columns for prediction",
        candidate_features,
        default=candidate_features,
    )

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        return

    data = df[feature_cols + [target_col]].dropna()
    if data.empty:
        st.warning("No rows remaining after dropping missing values. Please check your data.")
        return

    X = data[feature_cols]
    y = data[target_col]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    test_size = st.slider("Test size (for evaluation)", 0.1, 0.4, 0.2, step=0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if st.button("Train model"):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)

        left, right = st.columns(2)
        with left:
            st.metric("R² score", f"{r2:.4f}")
            st.metric("MAE", f"{mae:,.4f}")
            st.metric("RMSE", f"{rmse:,.4f}")
            if r2 >= 0.97:
                st.success("🎯 Model meets the 0.97+ R² target.")
            else:
                st.info(
                    "R² is below 0.97. You may experiment with different features, "
                    "data cleaning or feature engineering to improve performance."
                )

        with right:
            comp_df = pd.DataFrame(
                {"Actual": y_test.values, "Predicted": preds}, index=y_test.index
            )
            fig = px.scatter(comp_df, x="Actual", y="Predicted", trendline="ols")
            fig.update_layout(title="Actual vs Predicted – Adoption Score")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Feature importance (permutation)"):
            try:
                result = permutation_importance(
                    pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
                )
                pre = pipe.named_steps["preprocessor"]
                feature_names = pre.get_feature_names_out()
                importances = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": result["importances_mean"],
                    }
                ).sort_values("Importance", ascending=False)
                st.dataframe(importances.head(20).reset_index(drop=True))
            except Exception as e:
                st.warning(f"Could not compute feature importance: {e}")


def time_series_page(df):
    st.subheader("Time Series – Forecast Digital Transaction Volume")

    date_cols = [
        c
        for c in df.columns
        if np.issubdtype(df[c].dtype, np.datetime64)
        or "date" in c.lower()
        or "month" in c.lower()
        or "year" in c.lower()
    ]

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not date_cols:
        st.warning(
            "No obvious date-like columns detected. "
            "Please ensure your dataset has a date column (e.g., transaction_date)."
        )
        return
    if not num_cols:
        st.warning("No numeric transaction volume column found.")
        return

    date_col = st.selectbox("Select date column", date_cols)
    target_col = st.selectbox("Select transaction volume column", num_cols)

    horizon = st.slider("Forecast periods into the future", min_value=3, max_value=36, value=12)

    ts = df[[date_col, target_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col, target_col])
    ts = ts.sort_values(date_col)

    if ts.empty:
        st.warning("No valid time series data after cleaning. Please check your date and volume columns.")
        return

    ts["time_idx"] = np.arange(len(ts))
    X = ts[["time_idx"]]
    y = ts[target_col].values

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    ts["fitted"] = model.predict(X)

    inferred_freq = pd.infer_freq(ts[date_col])
    if inferred_freq is None:
        inferred_freq = "M"

    last_date = ts[date_col].iloc[-1]
    future_dates = pd.date_range(
        start=last_date, periods=horizon + 1, freq=inferred_freq
    )[1:]
    last_idx = ts["time_idx"].iloc[-1]
    future_idx = np.arange(last_idx + 1, last_idx + horizon + 1)
    future_preds = model.predict(future_idx.reshape(-1, 1))

    future_df = pd.DataFrame(
        {
            date_col: future_dates,
            target_col: future_preds,
            "type": "Forecast",
        }
    )
    history_df = ts[[date_col, target_col]].copy()
    history_df["type"] = "Actual"

    combined = pd.concat([history_df, future_df], ignore_index=True)

    fig = px.line(
        combined,
        x=date_col,
        y=target_col,
        color="type",
        title="Digital Transaction Volume – Actual & Forecast",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Sample forecast values:")
    st.dataframe(future_df.head())


def text_analytics_page(df):
    st.subheader("Text Analytics – Themes & Sentiment")

    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        st.warning("No text (object) columns detected for text analytics.")
        return

    text_col = st.selectbox("Select text column (e.g., media reports)", text_cols)
    n_topics = st.slider("Number of topics to extract", 2, 8, 3)

    texts = df[text_col].dropna().astype(str)
    if texts.empty:
        st.warning("No non-empty text found in the selected column.")
        return

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    X = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda")
    nmf.fit(X)
    feature_names = vectorizer.get_feature_names_out()

    st.markdown("### Key themes (topics)")
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_indices]
        st.write(f"**Topic {topic_idx + 1}:** " + ", ".join(top_words))

    if st.checkbox("Compute and view sentiment scores"):
        sent = texts.apply(lambda x: TextBlob(x).sentiment.polarity)
        sent_df = pd.DataFrame({"sentiment": sent})
        st.write("Sentiment polarity ranges from -1 (negative) to +1 (positive).")
        fig = px.histogram(sent_df, x="sentiment", nbins=30, title="Sentiment score distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sent_df.head())


def geo_dashboard_page(df):
    st.subheader("Geo Dashboard – Inclusion Index by Location")

    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    latitude_col = st.selectbox("Latitude column", df.columns, index=0 if lat_candidates == [] else df.columns.get_loc(lat_candidates[0]))
    longitude_col = st.selectbox("Longitude column", df.columns, index=0 if lon_candidates == [] else df.columns.get_loc(lon_candidates[0]))
    value_col = st.selectbox("Inclusion / adoption metric column", num_cols)

    label_col = st.selectbox(
        "Label column for tooltip (optional)", ["(none)"] + list(df.columns), index=0
    )

    geo_df = df[[latitude_col, longitude_col, value_col]].dropna()
    if label_col != "(none)":
        geo_df[label_col] = df[label_col]

    if geo_df.empty:
        st.warning("No rows with valid latitude, longitude and value.")
        return

    lat_mean = geo_df[latitude_col].mean()
    lon_mean = geo_df[longitude_col].mean()

    val = geo_df[value_col]
    if val.max() == val.min():
        scaled_radius = np.full_like(val, 5000, dtype=float)
    else:
        scaled_radius = 3000 + (val - val.min()) / (val.max() - val.min()) * 12000
    geo_df["radius"] = scaled_radius

    tooltip_text = "{%s}: {%s}" % (value_col, value_col)
    if label_col != "(none)":
        tooltip_text = "{%s}\\n%s" % (label_col, tooltip_text)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=geo_df,
        get_position=[longitude_col, latitude_col],
        get_radius="radius",
        get_fill_color=[30, 144, 255, 160],
        pickable=True,
    )

    view_state = pdk.ViewState(
        longitude=lon_mean,
        latitude=lat_mean,
        zoom=4,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": tooltip_text},
    )

    st.pydeck_chart(deck)

    st.write("Sample of geo data used:")
    st.dataframe(geo_df[[latitude_col, longitude_col, value_col]].head())


def main():
    st.title("Financial Inclusion Tracker – Analytics Prototype")

    st.sidebar.header("1. Upload dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Capstone dataset (CSV or Excel)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        df = load_file(uploaded_file)
        st.sidebar.success("Dataset loaded successfully.")
        st.session_state["df"] = df
    else:
        if "df" not in st.session_state:
            st.info("Please upload your dataset to begin.")
            return

    df = st.session_state["df"]

    st.sidebar.header("2. Navigate")
    page = st.sidebar.radio(
        "Select page",
        (
            "Overview",
            "ML – Adoption Score Prediction",
            "Time Series – Digital Transactions",
            "Text Analytics – Media Reports",
            "Geo Dashboard – Inclusion Index",
        ),
    )

    if page == "Overview":
        overview_page(df)
    elif page == "ML – Adoption Score Prediction":
        ml_page(df)
    elif page == "Time Series – Digital Transactions":
        time_series_page(df)
    elif page == "Text Analytics – Media Reports":
        text_analytics_page(df)
    elif page == "Geo Dashboard – Inclusion Index":
        geo_dashboard_page(df)


if __name__ == "__main__":
    main()
