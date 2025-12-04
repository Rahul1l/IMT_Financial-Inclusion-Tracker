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

st.set_page_config(page_title="UPI Adoption Prototype", layout="wide")

# --------------------------------------------------
# LOAD & MERGE ALL SHEETS COLUMN-WISE (NO KEY)
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


def merge_all_sheets_columnwise(sheets):
    df_final = None
    for name, df in sheets.items():
        df = df.copy().reset_index(drop=True)
        if df_final is None:
            df_final = df
        else:
            L = max(len(df_final), len(df))
            df_final = df_final.reindex(range(L)).reset_index(drop=True)
            df = df.reindex(range(L)).reset_index(drop=True)
            # Fix column collisions
            df.columns = [c if c not in df_final.columns else f"{c}_{name}" for c in df.columns]
            df_final = pd.concat([df_final, df], axis=1, join="outer")

    df_final = df_final.loc[:, ~df_final.columns.duplicated()]  # remove duplicate columns
    return df_final


def build_upi_adoption_score(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        df["UPI_Adoption_SScore"] = 50.0
        return df

    clean = df[numeric_cols].fillna(df[numeric_cols].median())
    X = StandardScaler().fit_transform(clean)
    comp = PCA(1, random_state=42).fit_transform(X).ravel()
    score = 50.0 if comp.max()==comp.min() else (comp-comp.min())/(comp.max()-comp.min())*100
    df["UPI_Adoption_SScore"] = score
    return df


# --------------------------------------------------
# ---------------------- PAGES ---------------------- #
# --------------------------------------------------

def page_overview(df):
    st.subheader("Dataset Overview")
    st.write(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]:,}")
    st.dataframe(df.head())


def page_ml(df):
    st.subheader("ML Model – Predict UPI Adoption Score")

    score_col = "UPI_Adoption_SScore"
    if score_col not in df.columns:
        st.error("❌ Adoption score column missing!")
        return

    X = df.drop(columns=[score_col], errors="ignore")
    y = df[score_col]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    transformer = make_column_transformer(
        (SimpleImputer(strategy='median'), num_cols),
        (SimpleImputer(strategy='most_frequent'), cat_cols),
        remainder="drop"
    )

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    pipe = make_pipeline(transformer, model)

    split = st.sidebar.slider("Test split", 0.1, 0.4, 0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split, random_state=42)

    if st.button("Train Model"):
        pipe.fit(X_tr, y_tr)
        pr = pipe.predict(X_te)

        r2 = r2_score(y_te, pr)
        mae = mean_absolute_error(y_te, pr)
        rmse = mean_squared_error(y_te, pr) ** 0.5

        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

        fig = px.scatter(pd.DataFrame({"Actual":y_te, "Predicted":pr}), x="Actual", y="Predicted", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)


def page_ts(df):
    st.subheader("Time Series Forecast")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns found for forecasting!")
        return

    vol_col = st.selectbox("Select transaction volume column", numeric_cols)

    ts = None
    date_col = None

    if date_cols:
        date_col = st.selectbox("Select Date column", date_cols)
        ts = df[[date_col, vol_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")

    elif "Year" in df.columns and "Month" in df.columns:
        y = st.selectbox("Select Year", ["Year"])
        m = st.selectbox("Select Month", ["Month"])
        ts = df[[y,m,vol_col]].copy()

        for c in [y,m,vol_col]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

        ts = ts.dropna(subset=[y,m,vol_col])

        if ts.empty:
            st.error("❌ No valid date rows left!")
            return

        ts["ts_date"] = pd.to_datetime(
            ts[y].astype(int).astype(str)+"-"+ts[m].astype(int).astype(str)+"-01",
            errors="coerce"
        )
        date_col = "ts_date"

    if ts is None:
        st.error("❌ No valid date or Year/Month structure found.")
        return

    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)

    if ts.empty:
        st.error("❌ No valid rows left for time series!")
        return

    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(300, random_state=42, n_jobs=-1).fit(ts[["t"]], ts[vol_col].values)

    future_steps = st.slider("Forecast months", 3, 24, 12)
    last = ts[date_col].iloc[-1]
    future_dates = pd.date_range(start=last, periods=future_steps+1, freq="M")[1:]
    future_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+future_steps)
    fut_preds = model.predict(future_t.reshape(-1,1))

    fut_df = pd.DataFrame({date_col:future_dates, vol_col:fut_preds, "type":"Forecast"})
    hist = ts[[date_col,vol_col]].copy(); hist["type"]="Actual"
    st.plotly_chart(px.line(pd.concat([hist,fut_df],ignore_index=True),x=date_col,y=vol_col,color="type"),use_container_width=True)
    st.dataframe(fut_df.head())


def page_text(df):
    st.subheader("Text Analytics + Sentiment")

    text_cols = df.select_dtypes(include="object").columns.tolist()

    if not text_cols:
        st.error("❌ No text columns found for NLP")
        return

    tcol = st.selectbox("Select text column for analytics", text_cols)
    data = df[tcol].dropna().astype(str)

    if data.empty:
        st.warning("⚠ No text rows available!")
        return

    num_topics = st.slider("Number of topics", 2, 6, 3)

    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
    tfidf = vectorizer.fit_transform(data)

    nmf = NMF(n_components=num_topics, random_state=42, init="nndsvda")
    nmf.fit(tfidf)
    words = vectorizer.get_feature_names_out()

    for i, t in enumerate(nmf.components_):
        topw = [words[idx] for idx in t.argsort()[-10:][::-1]]
        st.write(f"**Topic {i+1}:** " + ", ".join(topw))

    if st.checkbox("Run sentiment analysis"):
        data["sentiment"] = data.apply(lambda x: TextBlob(x).sentiment.polarity)
        st.plotly_chart(px.histogram(data["sentiment"], nbins=25, title="Sentiment Distribution"), use_container_width=True)
        st.dataframe(data[["sentiment"]].head())


def page_geo(df):
    st.subheader("Geo Dashboard – India States by UPI Adoption Score")

    if "State" not in df.columns:
        st.error("❌ `State` column not found in your dataset!")
        return

    geo = df.groupby("State", as_index=False)["UPI_Adoption_SScore"].mean()

    # Plot India state choropleth map
    fig = px.choropleth(
        geo,
        locations="State",
        locationmode="country names",
        color="UPI_Adoption_SScore",
        title="UPI Adoption Score Across India States",
        scope="asia",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(geo.head())


# --------------------------------------------------
# ---------------------- APP ROUTER ---------------- #
# --------------------------------------------------

def main():
    st.sidebar.title("Analytics Navigation")
    file = st.sidebar.file_uploader("Upload dataset", ["csv","xlsx"])

    if file:
        sheets = load_sheets(file)
        df = merge_all_sheets_columnwise(sheets)
        df = build_upi_adoption_score(df)
        st.session_state["df"] = df

    df = st.session_state.get("df")

    if df is None:
        st.title("Upload your capstone dataset to begin")
        return

    nav = st.sidebar.radio("Go to", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])
    if nav=="Overview":
        page_overview(df)
    elif nav=="ML Model":
        page_ml(df)
    elif nav=="Time Series":
        page_ts(df)
    elif nav=="Text Analytics":
        page_text(df)
    elif nav=="Geo Dashboard":
        page_geo(df)


if __name__ == "__main__":
    main()
