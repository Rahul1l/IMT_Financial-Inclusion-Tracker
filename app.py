import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import plotly.express as px
import pydeck as pdk

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
            df_final = pd.concat([df_final, df], axis=1, join="outer")
    # Fix duplicate column names if any
    df_final = df_final.loc[:, ~df_final.columns.duplicated(keep='first')]
    return df_final


# --------------------------------------------------
# BUILD TARGET: SYNTHETIC ADOPTION SCORE
# --------------------------------------------------

def build_upi_adoption_score(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        df["UPI_Adoption_Score"] = 50.0
        return df, []

    clean = df[num_cols].fillna(df[num_cols].median())
    X = StandardScaler().fit_transform(clean)
    comp = PCA(1, random_state=42).fit_transform(X).ravel()
    score = 50.0 if comp.max()==comp.min() else (comp-comp.min())/(comp.max()-comp.min())*100
    df["UPI_Adoption_Score"] = score
    return df, num_cols


# --------------------------------------------------
#  PAGES
# --------------------------------------------------

def page_overview(df):
    st.subheader("Dataset Overview")
    st.write(f"Total rows: {len(df):,}  |  Total columns: {len(df.columns):,}")
    st.dataframe(df.head())


def page_ml(df):
    st.subheader("ML Model – Predict UPI Adoption Score")

    if "UPI_Adoption_Score" not in df.columns:
        st.error("Target column missing!")
        return

    X = df.drop(columns=["UPI_Adoption_Score"], errors="ignore")
    y = df["UPI_Adoption_Score"]

    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(include="object").columns.tolist()

    transformer = make_column_transformer(
        (SimpleImputer(strategy="median"), num),
        (OneHotEncoder(handle_unknown="ignore"), cat),
        remainder="drop"
    )

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    pipe = make_pipeline(transformer, model)

    split = st.sidebar.slider("Test split %", 10, 40, 20) / 100
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split, random_state=42)

    if st.button("Train Model"):
        pipe.fit(X_tr, y_tr)
        pr = pipe.predict(X_te)

        r2 = r2_score(y_te, pr)
        mae = mean_absolute_error(y_te, pr)
        rmse = mean_squared_error(y_te, pr)**0.5

        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

        fig = px.scatter(pd.DataFrame({"Actual":y_te, "Predicted":pr}),
                         x="Actual", y="Predicted", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)


def page_ts(df):
    st.subheader("Time Series Forecast – Digital Transaction Volume")

    # Detect date formats safely
    date_cols = [c for c in df.columns if "date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]
    nums = df.select_dtypes(include=np.number).columns.tolist()

    if not nums:
        st.error("No numeric columns found!")
        return

    vol_col = st.selectbox("Select volume column", nums)
    ts = None
    date_col = None

    if date_cols:
        date_col = st.selectbox("Select date column", date_cols)
        ts = df[[date_col, vol_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    elif year_cols and month_cols:
        ycol = st.selectbox("Select year", year_cols)
        mcol = st.selectbox("Select month", month_cols)
        ts = df[[ycol, mcol, vol_col]].copy()
        for c in [ycol, mcol, vol_col]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")
        ts = ts.dropna(subset=[ycol, mcol, vol_col])
        if ts.empty:
            st.error("No valid rows!")
            return
        ts["ts_date"] = pd.to_datetime(
            ts[ycol].astype(int).astype(str) + "-" + ts[mcol].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        date_col = "ts_date"

    if ts is None:
        return

    date_col = date_col or "ts_date"
    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)
    if ts.empty:
        st.error("No valid time series rows left!")
        return

    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(300, random_state=42, n_jobs=-1)
    model.fit(ts[["t"]], ts[vol_col].values)

    steps = st.slider("Forecast months", 3, 24, 12)
    last = ts[date_col].iloc[-1]
    fut_dates = pd.date_range(start=last, periods=steps+1, freq="M")[1:]
    fut_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+steps)
    fut_pr = model.predict(fut_t.reshape(-1,1))

    fut_df = pd.DataFrame({date_col:fut_dates, vol_col:fut_pr, "type":"Forecast"})
    hist_df = ts[[date_col,vol_col]].copy()
    hist_df["type"]="Actual"

    st.plotly_chart(px.line(pd.concat([hist_df,fut_df],ignore_index=True),
                            x=date_col, y=vol_col, color="type"), use_container_width=True)
    st.dataframe(fut_df.head())


def page_text(df):
    st.subheader("Text Topic Extraction + Sentiment")

    txts = df.select_dtypes(include="object").columns.tolist()
    if not txts:
        st.warning("No text columns found")
        return

    col = st.selectbox("Pick text column", txts)
    data = df[col].dropna().astype(str)
    k = st.slider("Topics", 2, 6, 3)

    vec = TfidfVectorizer(max_features=1500, stop_words="english")
    X = vec.fit_transform(data)
    nmf = NMF(n_components=k, random_state=42, init="nndsvda").fit(X)
    words = vec.get_feature_names_out()

    for i, t in enumerate(nmf.components_):
        topw = [words[idx] for idx in t.argsort()[-10:][::-1]]
        st.write(f"Topic {i+1}: " + ", ".join(topw))

    if st.checkbox("Analyze sentiment"):
        score = data.apply(lambda x: TextBlob(x).sentiment.polarity)
        st.plotly_chart(px.histogram(score, nbins=25, title="Sentiment Distribution"), use_container_width=True)
        st.dataframe(score.head())


def page_geo(df):
    st.subheader("Geo Dashboard")

    score_col = "UPI_Adoption_Score"
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if not lat_cols or not lon_cols:
        st.error("No geo coordinates found")
        return

    label_col = st.selectbox("Pick label column", df.columns)
    lat = st.selectbox("Latitude", lat_cols)
    lon = st.selectbox("Longitude", lon_cols)

    geo = df[[label_col, lat, lon, score_col]].copy()
    for c in [lat,lon,score_col]:
        geo[c]=pd.to_numeric(geo[c],errors="coerce")
    geo = geo.dropna(subset=[lat,lon,score_col])

    if geo.empty:
        st.error("No valid geo rows left")
        return

    r = geo[score_col]
    geo["radius"] = 2000 + (r-r.min())/(r.max()-r.min()+1e-6)*8000
    layer = pdk.Layer("ScatterplotLayer",geo,get_position=[lon,lat],get_radius="radius",pickable=True)
    view = pdk.ViewState(latitude=geo[lat].mean(), longitude=geo[lon].mean(), zoom=4)

    st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=view,
                             tooltip={"text":f"{label_col}\nUPI Adoption Score: {{UPI_Adoption_Score}}"}))
    st.dataframe(geo.head())


# --------------------------------------------------
# APP ROUTER
# --------------------------------------------------

def main():
    file = st.sidebar.file_uploader("Upload dataset", ["csv","xlsx"])
    if file:
        sheets = load_sheets(file)
        df = merge_all_sheets_columnwise(sheets)
        df, _ = build_upi_adoption_score(df)
        st.session_state["df"]=df

    df = st.session_state.get("df")
    if df is None:
        st.title("Upload your capstone dataset")
        return

    nav = st.sidebar.radio("Navigate", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])
    if nav=="Overview": page_overview(df)
    elif nav=="Ml Model": page_ml(df)
    elif nav=="Time Series": page_ts(df)
    elif nav=="Text Analytics": page_text(df)
    elif nav=="Geo Dashboard": page_geo(df)


if __name__ == "__main__":
    main()
