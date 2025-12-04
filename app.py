import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from textblob import TextBlob
import plotly.express as px
import re

st.set_page_config(page_title="UPI Adoption Analytics", layout="wide")

# --------------------------------------------------
# PRE-CLEANING & HELPERS
# --------------------------------------------------

def sanitize_object_column(series: pd.Series) -> pd.Series:
    """Convert datetimes to string and ensure uniform str/numerics for encoders."""
    def _convert(x):
        if isinstance(x, (pd.Timestamp, np.datetime64)):
            return str(x)
        return x
    return series.apply(_convert)

# --------------------------------------------------
# 1. LOAD MULTI-SHEET EXCEL/CSV
# --------------------------------------------------

@st.cache_data
def load_sheets(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.title()
        return {"Main": df}
    else:
        xls = pd.ExcelFile(file)
        sheets = {s: xls.parse(s) for s in xls.sheet_names}
        for k in sheets:
            sheets[k].columns = sheets[k].columns.str.strip().str.title()
        return sheets

# --------------------------------------------------
# 2. MERGE ALL SHEETS COLUMN-WISE → OUTER JOIN ON INDEX
# --------------------------------------------------

def merge_all_sheets_columnwise(sheets):
    combined = None
    for name, df in sheets.items():
        df = df.copy().reset_index(drop=True)

        # Sanitize object columns to avoid ML encoder crashes
        for c in df.select_dtypes(include="object").columns:
            df[c] = sanitize_object_column(df[c]).astype(str)

        if combined is None:
            combined = df
        else:
            L = max(len(combined), len(df))
            combined = combined.reindex(range(L)).reset_index(drop=True)
            df = df.reindex(range(L)).reset_index(drop=True)

            # Avoid duplicate column names
            df.columns = [c if c not in combined.columns else f"{c}_{name}" for c in df.columns]
            combined = pd.concat([combined, df], axis=1, join="outer")

    # Drop exact duplicate columns if any slipped through
    combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    return combined

# --------------------------------------------------
# 3. CLEAN DATA → ENSURE NO MIXED DATETIME/STRING IN FEATURES
# --------------------------------------------------

def clean_data(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = sanitize_object_column(df[col]).astype(str)

        # Drop columns that are huge freeform text and unusable as ML features
        sample = df[col].dropna().head(30)
        if sample.str.len().mean() > 80 or df[col].nunique() > 300:
            df.drop(columns=[col], inplace=True)

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

    # Combine Year/Month into a TS-compatible date column if present
    y = [c for c in df.columns if "year" in c.lower()]
    m = [c for c in df.columns if "month" in c.lower()]
    if y and m and "ts_date" not in df.columns:
        try:
            df["ts_date"] = pd.to_datetime(
                df[y[0]].astype(int).astype(str)+"-"+df[m[0]].astype(int).astype(str)+"-01",
                errors="coerce"
            )
        except:
            df["ts_date"] = pd.NaT
    return df

# --------------------------------------------------
# 4. BUILD TARGET → ADOPTION SCORE FROM NUMERIC FACTOR COLUMNS
# --------------------------------------------------

def build_upi_adoption_score(df):
    df = clean_data(df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "UPI_Adoption_SScore"]

    if not numeric_cols:
        df["UPI_Adoption_SScore"] = 50.0
        return df

    X = StandardScaler().fit_transform(df[numeric_cols])
    comp = PCA(1, random_state=42).fit_transform(X).ravel()
    score = 50.0 if comp.max()==comp.min() else (comp-comp.min())/(comp.max()-comp.min())*100

    df["UPI_Adoption_SScore"] = score
    return df

# --------------------------------------------------
# 5. OVERVIEW PAGE
# --------------------------------------------------

def page_overview(df):
    st.subheader("Dataset Preview")
    st.write(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]:,}")
    st.dataframe(df.head())

# --------------------------------------------------
# 6. ML MODEL PAGE
# --------------------------------------------------

def page_ml(df):
    st.subheader("Machine Learning Model – State/District Level Adoption Prediction")

    target = "UPI_Adoption_SScore"
    if target not in df.columns:
        st.error("❌ Synthetic Adoption score column missing.")
        return

    X = df.drop(columns=[target], errors="ignore").copy()
    y = df[target]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # extra guard
    for c in cat_cols:
        X[c] = sanitize_object_column(X[c]).astype(str)

    transformer = make_column_transformer(
        (SimpleImputer(strategy="median"), num_cols),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols),
        remainder="drop"
    )

    model = RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1)
    pipe = make_pipeline(transformer, model)

    split = st.sidebar.slider("Test split (%)", 10, 40, 20) / 100
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split, random_state=42)

    if st.button("Train ML Model"):
        try:
            pipe.fit(X_tr, y_tr)
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            return

        pr = pipe.predict(X_te)
        r2 = r2_score(y_te, pr)
        mae = mean_absolute_error(y_te, pr)
        rmse = mean_squared_error(y_te, pr)**0.5

        st.metric("R² Accuracy", f"{r2*100:.2f}%")
        st.metric("MAE", round(mae, 4))
        st.metric("RMSE", round(rmse, 4))

        result_df = pd.DataFrame({"Actual": y_te, "Predicted": pr})
        fig = px.scatter(result_df, x="Actual", y="Predicted", trendline="ols", title="Actual vs Predicted Adoption Score")
        st.plotly_chart(fig, use_container_width=True)

        # Allow state-wise prediction view
        if "State" in df.columns:
            geo = df.groupby("State", as_index=False)[target].mean()
            X_all = df.drop(columns=[target]).copy()
            try:
                all_preds = pipe.predict(X_all)
                geo["Model_Predicted_Score"] = [
                    all_preds[df["State"] == s].mean() for s in geo["State"]
                ]
            except:
                geo["Model_Predicted_SScore"] = np.nan

            st.write("### Adoption Score by State (Predicted vs Actual Avg)")
            st.dataframe(geo.head())


# --------------------------------------------------
# 7. TIME SERIES FORECAST PAGE
# --------------------------------------------------

def page_ts(df):
    st.subheader("Time Series Forecasting – Digital Transaction Volume")

    date_cols = [c for c in df.columns if "date" in c.lower() or "ts_date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric volume column found for forecasting.")
        return

    vol_col = st.selectbox("Select Digital Transaction Volume Column", numeric_cols)

    ts = None
    date_col = None

    if date_cols:
        date_col = st.selectbox("Date Column", date_cols)
        ts = df[[date_col, vol_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")

    elif year_cols and month_cols:
        y = st.selectbox("Year", year_cols)
        m = st.selectbox("Month", month_cols)
        ts = df[[y, m, vol_col]].copy()
        for c in [y, m, vol_col]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")
        ts = ts.dropna(subset=[y, m, vol_col])
        if ts.empty:
            st.error("❌ No valid Year/Month rows left!")
            return
        ts["ts_date"] = pd.to_datetime(ts[y].astype(int).astype(str)+"-"+ts[m].astype(int).astype(str)+"-01", errors="coerce")
        date_col = "ts_date"

    if ts is None or date_col is None:
        st.error("❌ No usable date structure found!")
        return

    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)
    if ts.empty:
        st.error("No valid rows left after cleaning for time series!")
        return

    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(ts[["t"]], ts[vol_col])
    h = st.slider("Forecast months", 3, 36, 12)
    last = ts[date_col].iloc[-1]
    fut_dates = pd.date_range(start=last,periods=h+1,freq="M")[1:]
    fut_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+h)
    fut_pr = model.predict(fut_t.reshape(-1,1))
    fut_df = pd.DataFrame({date_col: fut_dates, vol_col:fut_pr, "Type":"Forecast"})
    hist_df = ts[[date_col, vol_col]].copy()
    hist_df["Type"]="Actual"
    st.plotly_chart(px.line(pd.concat([hist_df,fut_df],ignore_index=True),x=date_col,y=vol_col,color="Type",title="Actual vs Forecast"),use_container_width=True)
    st.dataframe(fut_df.head())

# --------------------------------------------------
# 8. TEXT ANALYTICS PAGE
# --------------------------------------------------

def page_text(df):
    st.subheader("Text Analytics – Topics + Sentiment")

    if "Report_Text" not in df.columns:
        st.error("❌ `Report_Text` column not found! Make sure the sheet is merged.")
        return

    texts = df["Report_Text"].dropna().astype(str)

    max_topics = st.sidebar.slider("Max topics", 2, 6, 3)
    vec = TfidfVectorizer(max_features=1800, stop_words="english")
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=max_topics, random_state=42, init="nndsvda")
    nmf.fit(X)
    words = vec.get_feature_names_out()

    for i, topic in enumerate(nmf.components_):
        top_words = [words[idx] for idx in topic.argsort()[-10:][::-1]]
        st.write(f"**Topic {i+1}:** " + ", ".join(top_words))

    if st.checkbox("Run sentiment analysis"):
        sentiment = texts.apply(lambda t: TextBlob(t).sentiment.polarity)
        st.plotly_chart(px.histogram(sentiment, nbins=25, title="Sentiment Distribution"))
        st.dataframe(sentiment.head())

# --------------------------------------------------
# 9. STATE GEO MAP PAGE
# --------------------------------------------------

def page_geo(df):
    st.subheader("Geo Dashboard – UPI Adoption Score by State")

    if "State" not in df.columns:
        st.error("❌ `State` column not found in dataset.")
        return

    metric = "UPI_Adoption_SScore"
    
    geo_df = df.groupby("State", as_index=False)[metric].mean()

    # interactive map where states are selectable
    fig = px.choropleth(
        geo_df,
        locations="State",
        locationmode="country names",
        color=metric,
        title="UPI Adoption Score Across Indian States",
        scope="asia"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("### Select state to view score:")
    chosen = st.selectbox("Choose state:", ["(none)"] + list(geo_df["State"]))
    if chosen != "(none)":
        sc = geo_df.loc[geo_df["State"]==chosen, metric].values[0]
        st.success(f"📍 {chosen} Selected!")
        st.metric("UPI Adoption Score", round(sc,4))

    st.write("### Adoption Score by State")
    st.dataframe(geo_df)

# --------------------------------------------------
# APP ROUTER
# --------------------------------------------------

def main():
    st.sidebar.header("Upload Dataset")
    file = st.sidebar.file_uploader("Upload file", ["csv","xlsx"])

    if file:
        sheets = load_sheets(file)
        df = merge_all_sheets_columnwise(sheets)
        df = build_upi_adoption_score(df)
        st.session_state["df"] = df

    df = st.session_state.get("df")

    if df is None:
        st.title("Upload dataset to begin")
        return

    nav = st.sidebar.radio("Navigation", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])
    if nav=="Overview": page_overview(df)
    elif nav=="ML Model": page_ml(df)
    elif nav=="Time Series": page_ts(df)
    elif nav=="Text Analytics": page_text(df)
    elif nav=="Geo Dashboard": page_geo(df)


if __name__ == "__main__":
    main()
