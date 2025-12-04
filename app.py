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
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance

from textblob import TextBlob
import plotly.express as px
import pydeck as pdk


st.set_page_config(
    page_title="Financial Inclusion Tracker – UPI Adoption",
    layout="wide",
)


# ---------------------- DATA LOADING & PREP ---------------------- #
@st.cache_data
def load_workbook(uploaded_file):
    """Load CSV / Excel. Always return a dict of {sheet_name: dataframe}."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return {"Main": df}
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheets = {}
        for sheet in xls.sheet_names:
            sheets[sheet] = xls.parse(sheet)
        return sheets


def combine_sheets_ui(sheets_dict):
    """
    Merge all sheets using a common key (if exists).
    This is where we use all 6 factor sheets together.
    """
    sheet_names = list(sheets_dict.keys())
    st.sidebar.write("Sheets found: " + ", ".join(sheet_names))

    dfs = [sheets_dict[name] for name in sheet_names]

    # Find common columns across all sheets to use as join key (e.g., District)
    common_keys = set(dfs[0].columns)
    for d in dfs[1:]:
        common_keys &= set(d.columns)

    if not common_keys or len(sheet_names) == 1:
        st.sidebar.info(
            "No common column across all sheets (or only one sheet). "
            "Using the first sheet as the combined dataset."
        )
        combined = dfs[0].copy()
        join_key = None
    else:
        common_keys = sorted(list(common_keys))
        join_key = st.sidebar.selectbox(
            "Join sheets using key column (common across all sheets)",
            common_keys,
        )
        combined = dfs[0].copy()
        for name in sheet_names[1:]:
            combined = combined.merge(
                sheets_dict[name],
                on=join_key,
                how="outer",
                suffixes=("", f"_{name}"),
            )

    return combined, join_key


def build_adoption_score(df):
    """
    Construct an artificial UPI Adoption Score using all numeric factors.
    This uses PCA on standardized numeric columns and scales to 0–100.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return df, None

    num_data = df[num_cols].copy()
    num_data = num_data.fillna(num_data.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_data)

    pca = PCA(n_components=1, random_state=42)
    component = pca.fit_transform(X_scaled).ravel()

    if component.max() == component.min():
        score = np.full_like(component, 50.0, dtype=float)
    else:
        score = (component - component.min()) / (component.max() - component.min()) * 100

    df_with_score = df.copy()
    df_with_score["Adoption_Score"] = score

    return df_with_score, num_cols


# ---------------------- PAGES ---------------------- #
def overview_page(sheets_dict, combined_df, join_key):
    st.subheader("Dataset Overview – All UPI Adoption Factors")

    tabs = st.tabs(["Combined Dataset"] + list(sheets_dict.keys()))

    # Combined view
    with tabs[0]:
        st.markdown("**Combined dataset (all factor sheets merged)**")
        if join_key:
            st.write(f"Joined using key column: `{join_key}`")
        else:
            st.write("No common join key – first sheet used as combined dataset.")
        st.write(f"Rows: {combined_df.shape[0]:,}  |  Columns: {combined_df.shape[1]:,}")
        st.dataframe(combined_df.head())

        with st.expander("Combined dataset column information"):
            info_df = pd.DataFrame(
                {
                    "column": combined_df.columns,
                    "dtype": combined_df.dtypes.astype(str).values,
                    "non_nulls": combined_df.notnull().sum().values,
                    "nulls": combined_df.isnull().sum().values,
                }
            )
            st.dataframe(info_df)

    # Individual sheets
    for i, (sheet_name, df) in enumerate(sheets_dict.items(), start=1):
        with tabs[i]:
            st.markdown(f"**Sheet: {sheet_name}**")
            st.write(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]:,}")
            st.dataframe(df.head())


def ml_page(combined_df):
    st.subheader("ML Model – UPI Adoption Score Prediction (across districts)")

    if "Adoption_Score" not in combined_df.columns:
        st.warning("Adoption_Score column missing. Please re-upload your dataset.")
        return

    target_col = "Adoption_Score"

    candidate_features = [c for c in combined_df.columns if c != target_col]
    st.write(
        "The app has created an **Adoption_Score** (0–100) using all numeric factors "
        "from your six sheets via PCA. We now build an ML model to predict this score."
    )

    feature_cols = st.multiselect(
        "Select feature columns (factors) to use in the prediction model",
        candidate_features,
        default=candidate_features,
    )

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        return

    data = combined_df[feature_cols + [target_col]].dropna()
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

    if st.button("Train adoption score model"):
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
                    "R² is below 0.97. Try adding/removing factors or improving data quality."
                )

        with right:
            comp_df = pd.DataFrame(
                {"Actual": y_test.values, "Predicted": preds}, index=y_test.index
            )
            fig = px.scatter(comp_df, x="Actual", y="Predicted", trendline="ols")
            fig.update_layout(title="Actual vs Predicted – UPI Adoption Score")
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
                st.dataframe(importances.head(25).reset_index(drop=True))
            except Exception as e:
                st.warning(f"Could not compute feature importance: {e}")


def time_series_page(combined_df):
    st.subheader("Time Series – Forecast Digital Transaction Volume")

    # Work on combined dataset (which already merges all factors)
    df = combined_df

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
            "No obvious date-like columns detected in the combined dataset. "
            "Please ensure at least one sheet has a proper date column."
        )
        return
    if not num_cols:
        st.warning("No numeric column found to forecast.")
        return

    date_col = st.selectbox("Select date column", date_cols)
    target_col = st.selectbox("Select transaction volume column", num_cols)

    horizon = st.slider("Forecast periods into the future", min_value=3, max_value=36, value=12)

    ts = df[[date_col, target_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col, target_col])
    ts = ts.sort_values(date_col)

    if ts.empty:
        st.warning("No valid time series data after cleaning. Please check your date & volume columns.")
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
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=inferred_freq)[1:]
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

    combined_ts = pd.concat([history_df, future_df], ignore_index=True)

    fig = px.line(
        combined_ts,
        x=date_col,
        y=target_col,
        color="type",
        title="Digital Transaction Volume – Actual & Forecast",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Sample forecast values:")
    st.dataframe(future_df.head())


def text_analytics_page(sheets_dict, combined_df):
    st.subheader("Text Analytics – Media / Policy Reports")

    # Choose which dataset to use (some sheets may be pure text)
    options = ["Combined"] + list(sheets_dict.keys())
    choice = st.selectbox("Select data source for text analytics", options)

    df = combined_df if choice == "Combined" else sheets_dict[choice]

    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        st.warning("No text (object) columns detected in the selected sheet.")
        return

    text_col = st.selectbox("Select text column", text_cols)
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


def geo_dashboard_page(combined_df):
    st.subheader("Geo Dashboard – UPI Inclusion / Adoption Index by Location")

    df = combined_df

    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.warning("No numeric columns available for inclusion/adoption value.")
        return

    latitude_col = st.selectbox(
        "Latitude column",
        df.columns,
        index=df.columns.get_loc(lat_candidates[0]) if lat_candidates else 0,
    )
    longitude_col = st.selectbox(
        "Longitude column",
        df.columns,
        index=df.columns.get_loc(lon_candidates[0]) if lon_candidates else 0,
    )
    value_col = st.selectbox(
        "Inclusion / adoption metric column (e.g., Adoption_Score)",
        num_cols,
        index=num_cols.index("Adoption_Score") if "Adoption_Score" in num_cols else 0,
    )
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


# ---------------------- MAIN ---------------------- #
def main():
    st.title("Financial Inclusion Tracker – UPI Adoption Prototype")

    st.sidebar.header("1. Upload multi-sheet dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Capstone workbook (Excel/CSV with 6 factor sheets)",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        sheets_dict = load_workbook(uploaded_file)
        st.session_state["sheets_dict"] = sheets_dict
    elif "sheets_dict" not in st.session_state:
        st.info("Please upload your multi-sheet dataset to begin.")
        return

    sheets_dict = st.session_state["sheets_dict"]

    st.sidebar.header("2. Combine sheets & build adoption score")
    combined_df, join_key = combine_sheets_ui(sheets_dict)
    combined_df, _ = build_adoption_score(combined_df)
    st.session_state["combined_df"] = combined_df

    st.sidebar.header("3. Navigate")
    page = st.sidebar.radio(
        "Select page",
        (
            "Overview",
            "ML – Adoption Score Prediction",
            "Time Series – Digital Transactions",
            "Text Analytics",
            "Geo Dashboard – Inclusion Index",
        ),
    )

    combined_df = st.session_state["combined_df"]

    if page == "Overview":
        overview_page(sheets_dict, combined_df, join_key)
    elif page == "ML – Adoption Score Prediction":
        ml_page(combined_df)
    elif page == "Time Series – Digital Transactions":
        time_series_page(combined_df)
    elif page == "Text Analytics":
        text_analytics_page(sheets_dict, combined_df)
    elif page == "Geo Dashboard – Inclusion Index":
        geo_dashboard_page(combined_df)


if __name__ == "__main__":
    main()
