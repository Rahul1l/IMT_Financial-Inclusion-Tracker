# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from textblob import TextBlob
import plotly.express as px
import pydeck as pdk


st.set_page_config(layout="wide", page_title="UPI Adoption Tracker")

# ---- Load all sheets ----
@st.cache_data
def load_sheets(file):
    xls = pd.ExcelFile(file)
    return {s: xls.parse(s) for s in xls.sheet_names}

# ---- Merge sheets by index, outer join, all columns retained ----
def merge_sheets_columnwise(sheets):
    df_final = None
    for name, df in sheets.items():
        df = df.copy().reset_index(drop=True)
        df.columns = [c.strip() for c in df.columns]  # strip spaces
        if df_final is None:
            df_final = df
        else:
            max_len = max(len(df_final), len(df))
            df_final = df_final.reindex(range(max_len))
            df = df.reindex(range(max_len))
            df_final = pd.concat([df_final, df], axis=1)
    return df_final


# ---- Build synthetic Adoption Score from numeric factors ----
def compute_upi_score(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        df["UPI_Adoption_Score"] = 50.0
        return df, []
    clean = df[num_cols].fillna(df[num_cols].median())
    X = StandardScaler().fit_transform(clean)
    pca = PCA(1, random_state=42)
    comp = pca.fit_transform(X).ravel()
    score = 50.0 if comp.max() == comp.min() else (comp - comp.min())/(comp.max()-comp.min())*100
    df["UPI_Adoption_Score"] = score
    return df, num_cols


# ---- ML Page ----
def page_ml(df):
    st.subheader("ML Model: Predict Computed UPI Adoption Score")

    target = "UPI_Adoption_Score"
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([("i", SimpleImputer("most_frequent")),
                              ("o", OneHotEncoder(handle_unknown="ignore"))]),
             cat_cols)
        ], remainder="drop"
    )

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    pipe = Pipeline([("p", pre), ("m", model)])

    split = st.sidebar.slider("Test split", 0.1, 0.4, 0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split, random_state=42)

    if st.button("Train model"):
        try:
            pipe.fit(X_tr, y_tr)
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            st.stop()

        pr = pipe.predict(X_te)
        r2 = r2_score(y_te, pr)
        mae = mean_absolute_error(y_te, pr)
        rmse = mean_squared_error(y_te, pr)**0.5

        st.metric("R²", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

        fig = px.scatter(pd.DataFrame({"Actual":y_te, "Predicted":pr}),
                         x="Actual", y="Predicted", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)


# ---- Time Series Forecast Page ----
def page_ts(df):
    st.subheader("Time Series Forecast: Digital Transaction Volume")

    # detect date patterns
    date_cols = [c for c in df.columns if "date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.error("❌ No numeric columns found for forecasting!")
        return

    vol_col = st.selectbox("Select transaction volume column", num_cols)

    if date_cols:
        date_col = st.selectbox("Select date column", date_cols)
        ts = df[[date_col, vol_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")

    elif year_cols and month_cols:
        y = st.selectbox("Select year column", year_cols)
        m = st.selectbox("Select month column", month_cols)
        ts = df[[y, m, vol_col]].copy()

        for c in [y, m, vol_col]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

        ts = ts.dropna(subset=[y, m, vol_col])
        if ts.empty:
            st.error("❌ No valid Year+Month rows left to assemble time series!")
            return

        ts["ts_date"] = pd.to_datetime(
            ts[y].astype(int).astype(str) + "-" + ts[m].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        date_col = "ts_date"
    else:
        st.error("❌ No valid date structure found.")
        return

    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)
    if ts.empty:
        st.error("❌ No valid rows left after datetime conversion!")
        return

    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(300, random_state=42, n_jobs=-1)
    model.fit(ts[["t"]], ts[vol_col].values)

    steps = st.slider("Forecast months", 3, 36, 12)
    fut_dates = pd.date_range(start=ts[date_col].iloc[-1], periods=steps+1, freq="M")[1:]
    fut_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+steps)
    fut_pr = model.predict(fut_t.reshape(-1,1))

    fut_df = pd.DataFrame({date_col:fut_dates, vol_col:fut_pr, "type":"Forecast"})
    hist_df = ts[[date_col, vol_col]].copy()
    hist_df["type"]="Actual"

    fig = px.line(pd.concat([hist_df, fut_df], ignore_index=True), x=date_col, y=vol_col, color="type")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fut_df.head())


# ---- Text Analytics ----
def page_text(df):
    st.subheader("Sentiment + Topics on District Reports")

    text_cols = df.select_dtypes(include="object").columns.tolist()
    if not text_cols:
        st.warning("⚠ No text columns found for NLP.")
        return

    col = st.selectbox("Pick text column", text_cols)
    data = df[col].dropna().astype(str)

    k = st.slider("Topics", 2, 6, 3)
    vec = TfidfVectorizer(max_features=1500, stop_words="english")
    X = vec.fit_transform(data)

    nmf = NMF(n_components=k,random_state=42,init="nndsvda").fit(X)
    words = vec.get_feature_names_out()

    for i, t in enumerate(nmf.components_):
        topw = [words[idx] for idx in t.argsort()[-10:][::-1]]
        st.write(f"**Topic {i+1}:** " + ", ".join(topw))

    if st.checkbox("Analyze sentiment"):
        score = data.apply(lambda x: TextBlob(x).sentiment.polarity)
        st.plotly_chart(px.histogram(score, nbins=30, title="Sentiment Distribution"), use_container_width=True)
        st.dataframe(score.head())


# ---- Geo Dashboard ----
def page_geo(df):
    st.subheader("Geo Dashboard: UPI Adoption Score")

    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if not lat_cols or not lon_cols:
        st.error("❌ No lat/lon columns found!")
        return

    district_col = st.selectbox("Select district label column", df.columns)

    lat_col = st.selectbox("Latitude column", lat_cols)
    lon_col = st.selectbox("Longitude column", lon_cols)

    geo = df[[district_col, lat_col, lon_col, "UPI_Adoption_Score"]].copy()
    for c in [lat_col, lon_col, "UPI_Adoption_Score"]:
        geo[c] = pd.to_numeric(geo[c], errors="coerce")

    geo = geo.dropna()
    if geo.empty:
        st.error("❌ No valid geo rows left!")
        return

    r = geo["UPI_Adoption_Score"]
    geo["radius"] = 2000 + (r-r.min())/(r.max()-r.min()+1e-5)*8000

    layer = pdk.Layer("ScatterplotLayer", geo, get_position=[lon_col, lat_col], get_radius="radius", pickable=True)
    view = pdk.ViewState(latitude=geo[lat_col].mean(), longitude=geo[lon_col].mean(), zoom=4)

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"UPI_Adoption_Score: {UPI_Adoption_Score}"}))
    st.dataframe(geo.head())


# --------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------

def main():
    st.sidebar.header("Upload Dataset")
    file = st.sidebar.file_uploader("Upload capstone file", ["csv","xlsx"])

    if file:
        sheets = load_sheets(file)
        df = merge_sheets_columnwise(sheets)
        df, _ = compute_upi_score(df)
        st.session_state["df"]=df

    if "df" not in st.session_state:
        st.title("Upload your dataset to begin")
        return

    df = st.session_state["df"]

    nav = st.sidebar.radio("Navigate", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])
    if nav=="Overview":
        page_overview(df, None)
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
