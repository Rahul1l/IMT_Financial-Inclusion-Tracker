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
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from textblob import TextBlob
import plotly.express as px
import pydeck as pdk


st.set_page_config(
    page_title="UPI Adoption Financial Tracker",
    layout="wide",
)

# --------------------------------------------------
# 1. LOADING & COMBINING SHEETS (NO COMMON KEY)
# --------------------------------------------------

@st.cache_data
def load_sheets(file):
    """
    Load CSV/Excel and return dict of {sheet_name: DataFrame}.
    For CSV, treat as a single sheet.
    """
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
        return {"Main": df}
    else:
        xls = pd.ExcelFile(file)
        sheets = {}
        for s in xls.sheet_names:
            sheets[s] = xls.parse(s)
        return sheets


def combine_sheets_by_columns(sheets_dict):
    """
    Combine all sheets using outer join by columns (no common key).
    We align rows by position (index) and concatenate columns.
    If a column name would collide, we suffix it with the sheet name.
    """
    combined = None

    for sheet_name, df in sheets_dict.items():
        temp = df.copy().reset_index(drop=True)

        # Avoid duplicate column names across sheets
        new_cols = []
        existing = set() if combined is None else set(combined.columns)
        for c in temp.columns:
            new_name = c
            # If already exists, add sheet suffix
            if new_name in existing or new_name in new_cols:
                new_name = f"{c}_{sheet_name}"
            new_cols.append(new_name)
        temp.columns = new_cols

        if combined is None:
            combined = temp
        else:
            # Align length
            max_len = max(len(combined), len(temp))
            combined = combined.reindex(range(max_len)).reset_index(drop=True)
            temp = temp.reindex(range(max_len)).reset_index(drop=True)
            combined = pd.concat([combined, temp], axis=1)

    return combined


def build_upi_adoption_score(df):
    """
    Construct UPI_Adoption_Score (0-100) from all numeric columns using PCA.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        # Fallback: constant score if no numeric data
        df["UPI_Adoption_Score"] = 50.0
        return df, []

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

    out = df.copy()
    out["UPI_Adoption_Score"] = score

    return out, num_cols


# --------------------------------------------------
# 2. PAGES
# --------------------------------------------------

def page_overview(df):
    st.subheader("Dataset Overview – Combined All Sheets (Column-wise)")
    st.write(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]:,}")
    st.dataframe(df.head())

    with st.expander("Column information"):
        info = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_nulls": df.notnull().sum().values,
            "nulls": df.isnull().sum().values,
        })
        st.dataframe(info)


def page_ml_model(df):
    st.subheader("ML Model – Predict UPI Adoption Score")

    if "UPI_Adoption_Score" not in df.columns:
        st.error("UPI_Adoption_Score column is missing.")
        return

    target_col = "UPI_Adoption_Score"
    feature_cols = [c for c in df.columns if c != target_col]

    data = df[feature_cols + [target_col]].dropna()
    if data.empty:
        st.warning("No rows left after dropping missing values.")
        return

    X = data[feature_cols]
    y = data[target_col]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

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

    test_size = st.slider("Test size (evaluation)", 0.1, 0.4, 0.2, step=0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if st.button("Train UPI Adoption Model"):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5  # no squared= for compatibility

        col1, col2, col3 = st.columns(3)
        col1.metric("R² score", f"{r2:.4f}")
        col2.metric("MAE", f"{mae:,.4f}")
        col3.metric("RMSE", f"{rmse:,.4f}")

        if r2 >= 0.97:
            st.success("🎯 Model meets the 0.97+ R² target.")
        else:
            st.info("R² < 0.97. Try cleaning data or adjusting features.")

        comp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
        fig = px.scatter(comp_df, x="Actual", y="Predicted", trendline="ols",
                         title="Actual vs Predicted – UPI Adoption Score")
        st.plotly_chart(fig, use_container_width=True)


def page_time_series(df):
    st.subheader("Time Series – Forecast Digital Transaction Volume")

    # Identify potential date parts
    date_cols = [c for c in df.columns if "date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.error("No numeric columns available for forecasting.")
        return

    vol_col = st.selectbox("Select transaction volume column", num_cols)

    ts = None
    date_col = None

    # Case 1: full date column
    if date_cols:
        date_col = st.selectbox("Select date column", date_cols)
        ts = df[[date_col, vol_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")

    # Case 2: year + month
    elif year_cols and month_cols:
        ycol = st.selectbox("Select Year column", year_cols)
        mcol = st.selectbox("Select Month column", month_cols)

        ts = df[[ycol, mcol, vol_col]].copy()
        ts[ycol] = pd.to_numeric(ts[ycol], errors="coerce")
        ts[mcol] = pd.to_numeric(ts[mcol], errors="coerce")
        ts = ts.dropna(subset=[ycol, mcol, vol_col])

        ts["combined_date"] = pd.to_datetime(
            ts[ycol].astype(int).astype(str) + "-" +
            ts[mcol].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        date_col = "combined_date"

    else:
        st.error("No usable date / (year + month) columns found.")
        return

    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)
    if ts.empty:
        st.error("No valid rows left for time series after cleaning.")
        return

    ts["t"] = np.arange(len(ts))
    X = ts[["t"]]
    y = ts[vol_col].values

    model = RandomForestRegressor(
        n_estimators=350, random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    steps = st.slider("Forecast periods (months)", 3, 36, 12)
    freq = "M"
    last_date = ts[date_col].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
    future_t = np.arange(ts["t"].iloc[-1] + 1, ts["t"].iloc[-1] + 1 + steps)
    future_preds = model.predict(future_t.reshape(-1, 1))

    future_df = pd.DataFrame(
        {date_col: future_dates, vol_col: future_preds, "type": "Forecast"}
    )
    history_df = ts[[date_col, vol_col]].copy()
    history_df["type"] = "Actual"

    combined = pd.concat([history_df, future_df], ignore_index=True)

    fig = px.line(
        combined,
        x=date_col,
        y=vol_col,
        color="type",
        title="Digital Transaction Volume – Actual & Forecast",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Forecast preview:")
    st.dataframe(future_df.head())


def page_text_analytics(df):
    st.subheader("Text Analytics – Themes & Sentiment")

    text_cols = df.select_dtypes(include="object").columns.tolist()
    if not text_cols:
        st.warning("No text columns (object dtype) found.")
        return

    text_col = st.selectbox("Select a text column", text_cols)
    texts = df[text_col].dropna().astype(str)

    if texts.empty:
        st.warning("No non-empty text rows in the selected column.")
        return

    n_topics = st.slider("Number of topics", 2, 8, 3)

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    X = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda")
    nmf.fit(X)
    feature_names = vectorizer.get_feature_names_out()

    st.markdown("### Key Themes (Topics)")
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_indices]
        st.write(f"**Topic {topic_idx + 1}:** " + ", ".join(top_words))

    if st.checkbox("Compute and show sentiment distribution"):
        sentiment = texts.apply(lambda x: TextBlob(x).sentiment.polarity)
        sent_df = pd.DataFrame({"sentiment": sentiment})
        fig = px.histogram(sent_df, x="sentiment", nbins=30,
                           title="Sentiment Score Distribution (-1 to 1)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sent_df.head())


def page_geo_dashboard(df):
    st.subheader("Geo Dashboard – UPI Adoption Score by Location")

    if "UPI_Adoption_Score" not in df.columns:
        st.error("UPI_Adoption_Score column is missing for geo visualization.")
        return

    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if not lat_candidates or not lon_candidates:
        st.error("No Latitude/Longitude-like columns detected.")
        return

    latitude_col = st.selectbox("Latitude column", lat_candidates)
    longitude_col = st.selectbox("Longitude column", lon_candidates)
    label_col = st.selectbox("Label column (district name, etc.)",
                             ["(none)"] + list(df.columns),
                             index=0)

    geo_df = df[[latitude_col, longitude_col, "UPI_Adoption_Score"]].copy()

    # Force numeric for lat/lon/value
    for col in [latitude_col, longitude_col, "UPI_Adoption_Score"]:
        geo_df[col] = pd.to_numeric(geo_df[col], errors="coerce")

    geo_df = geo_df.dropna(subset=[latitude_col, longitude_col, "UPI_Adoption_Score"])

    if label_col != "(none)":
        geo_df[label_col] = df[label_col]

    if geo_df.empty:
        st.error("No valid rows with numeric lat/lon and adoption score.")
        return

    # Map center
    lat_mean = geo_df[latitude_col].mean()
    lon_mean = geo_df[longitude_col].mean()

    # Scale radius by adoption score
    val = geo_df["UPI_Adoption_Score"]
    if val.max() == val.min():
        geo_df["radius"] = 5000.0
    else:
        geo_df["radius"] = 3000 + (val - val.min()) / (val.max() - val.min()) * 12000

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=geo_df,
        get_position=[longitude_col, latitude_col],
        get_radius="radius",
        get_fill_color=[30, 144, 255, 160],
        pickable=True,
    )

    tooltip_text = "UPI_Adoption_Score: {UPI_Adoption_Score}"
    if label_col != "(none)":
        tooltip_text = f"{label_col}: {{{label_col}}}\n" + tooltip_text

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
    st.dataframe(geo_df[[latitude_col, longitude_col, "UPI_Adoption_Score"]].head())


# --------------------------------------------------
# 3. MAIN
# --------------------------------------------------

def main():
    st.title("UPI Adoption – Financial Inclusion Analytics Prototype")

    st.sidebar.header("1. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Capstone workbook (Excel with multiple sheets or CSV)",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        sheets = load_sheets(uploaded_file)
        combined = combine_sheets_by_columns(sheets)
        combined, _ = build_upi_adoption_score(combined)
        st.session_state["df"] = combined

    if "df" not in st.session_state:
        st.info("Please upload your multi-sheet dataset to start.")
        return

    df = st.session_state["df"]

    st.sidebar.header("2. Navigate")
    page = st.sidebar.radio(
        "Select page",
        ["Overview", "ML Model", "Time Series", "Text Analytics", "Geo Dashboard"],
    )

    if page == "Overview":
        page_overview(df)
    elif page == "ML Model":
        page_ml_model(df)
    elif page == "Time Series":
        page_time_series(df)
    elif page == "Text Analytics":
        page_text_analytics(df)
    elif page == "Geo Dashboard":
        page_geo_dashboard(df)


if __name__ == "__main__":
    main()
