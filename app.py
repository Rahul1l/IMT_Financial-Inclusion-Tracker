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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from textblob import TextBlob
import plotly.express as px
import pydeck as pdk


st.set_page_config(layout="wide", page_title="UPI Adoption Tracker")

# --------------------------------------------------
# LOAD & COMBINE ALL SHEETS (COLUMN-WISE)
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
    combined = None
    for name, df in sheets.items():
        df = df.copy().reset_index(drop=True)
        df = df.reindex(range(len(df)))  # ensure index alignment
        if combined is None:
            combined = df
        else:
            max_len = max(len(combined), len(df))
            combined = combined.reindex(range(max_len))
            df = df.reindex(range(max_len))
            # Suffix column names if collision
            df.columns = [c if c not in combined.columns else f"{c}_{name}" for c in df.columns]
            combined = pd.concat([combined, df], axis=1)
    return combined


# --------------------------------------------------
# COMPUTE SYNTHETIC UPI ADOPTION SCORE USING PCA
# --------------------------------------------------

def build_upi_adoption_score(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        df["UPI_Adoption_Score"] = 50.0
        return df

    clean = df[numeric_cols].fillna(df[numeric_cols].median())
    X = StandardScaler().fit_transform(clean)
    comp = PCA(1, random_state=42).fit_transform(X).ravel()
    
    if comp.max() == comp.min():
        score = 50.0
    else:
        score = (comp - comp.min()) / (comp.max() - comp.min()) * 100

    df["UPI_Adoption_Score"] = score
    return df


# --------------------------------------------------
# ML MODEL PAGE
# --------------------------------------------------

def page_ml(df):
    if "UPI_Adoption_Score" not in df.columns:
        st.error("Score column missing! Re-upload dataset.")
        return

    target = "UPI_Adoption_Score"
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()

    transformer = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer("median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer("most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ], remainder="drop")

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    pipe = Pipeline([("p", transformer), ("m", model)])

    split = st.sidebar.slider("Test split", 0.1, 0.5, 0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split, random_state=42)

    if st.button("Train model"):
        pipe.fit(X_tr, y_tr)
        pr = pipe.predict(X_te)
        st.metric("R²", f"{r2_score(y_te, pr):.4f}")
        st.metric("MAE", f"{mean_absolute_error(y_te, pr):.4f}")
        st.metric("RMSE", f"{mean_squared_error(y_te, pr)**0.5:.4f}")

        st.plotly_chart(px.scatter(pd.DataFrame({"Actual":y_te, "Predicted":pr}), x="Actual", y="Predicted", trendline="ols"), use_container_width=True)


# --------------------------------------------------
# TIME SERIES FORECAST PAGE
# --------------------------------------------------

def page_ts(df):
    date_cols = [c for c in df.columns if "date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    vol = st.selectbox("Transaction Volume Column", num_cols)

    ts = None
    date_col = None

    if "district" in [c.lower() for c in df.columns]:
        st.write("⚠ Forecast ignores District column (no common key present)")

    if date_cols:
        date_col = st.selectbox("Date Column", date_cols)
        ts = df[[date_col, vol]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        date_col = date_col

    elif year_cols and month_cols:
        y = st.selectbox("Year", year_cols)
        m = st.selectbox("Month", month_cols)
        ts = df[[y, m, vol]].copy()
        for c in [y, m, vol]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")
        ts = ts.dropna(subset=[y, m, vol])

        if ts.empty:
            st.error("No valid rows left to form time series")
            return

        ts["ts_date"] = pd.to_datetime(
            ts[y].astype(int).astype(str) + "-" + ts[m].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        date_col = "ts_date"

    else:
        st.error("No valid date structure found")
        return

    ts = ts.dropna(subset=[date_col, vol]).sort_values(date_col)
    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(300, random_state=42, n_jobs=-1).fit(ts[["t"]], ts[vol].values)

    h = st.slider("Forecast months", 3, 36, 12)
    fut_dates = pd.date_range(start=ts[date_col].iloc[-1], periods=h+1, freq="M")[1:]
    fut_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+h)
    fut_pr = model.predict(fut_t.reshape(-1,1))

    fut_df = pd.DataFrame({date_col:fut_dates,vol:fut_pr,"type":"Forecast"})
    hist = ts[[date_col,vol]].copy(); hist["type"]="Actual"
    st.plotly_chart(px.line(pd.concat([hist,fut_df],ignore_index=True),x=date_col,y=vol,color="type"),use_container_width=True)
    st.dataframe(fut_df.head())


# --------------------------------------------------
# TEXT ANALYTICS PAGE
# --------------------------------------------------

def page_text(df):
    txts = df.select_dtypes(include="object").columns.tolist()
    if not txts:
        st.warning("No text columns found")
        return
    col = st.selectbox("Text column", txts)
    data = df[col].dropna().astype(str)

    k = st.slider("Topics", 2, 6, 3)
    vec = TfidfVectorizer(max_features=1500, stop_words="english")
    X = vec.fit_transform(data)
    nmf = NMF(n_components=k,random_state=42,init="nndsvda").fit(X)
    words = vec.get_feature_names_out()

    for i, t in enumerate(nmf.components_):
        topw = [words[idx] for idx in t.argsort()[-10:][::-1]]
        st.write(f"Topic {i+1}: " + ", ".join(topw))

    if st.checkbox("Sentiment"):
        score = data.apply(lambda x: TextBlob(x).sentiment.polarity)
        st.plotly_chart(px.histogram(score, nbins=25, title="Sentiment Distribution"))
        st.dataframe(score.head())


# --------------------------------------------------
# GEO DASHBOARD PAGE
# --------------------------------------------------

def page_geo(df):
    if "UPI_Adoption_SScore" not in df.columns:
        st.error("No adoption score column available for map!")
        return

    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if not lat_cols or not lon_cols:
        st.error("No latitude/longitude columns found!")
        return

    label = st.selectbox("District/Label", df.columns)
    lat = st.selectbox("Latitude", lat_cols)
    lon = st.selectbox("Longitude", lon_cols)

    geo_df = df[[label, lat, lon, "UPI_Adoption_SScore"]].copy()
    for c in [lat, lon, "UPI_Adoption_SScore"]:
        geo[c] = pd.to_numeric(geo[c], errors="coerce")

    geo_df = geo_df.dropna()
    if geo_df.empty:
        st.error("No valid geo rows left after cleaning")
        return

    r = geo_df["UPI_Adoption_SScore"]
    geo_df["radius"] = 2000 + (r - r.min())/(r.max()-r.min()+1e-5)*8000
    layer = pdk.Layer("ScatterplotLayer", geo_df, get_position=[lon, lat], get_radius="radius", pickable=True)
    view = pdk.ViewState(latitude=geo_df[lat_col].mean(), longitude=geo_df[lon_col].mean(), zoom=4)

    st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=view,tooltip={"text":"UPI Adoption: {UPI_Adoption_SScore}"}))
    st.dataframe(geo_df.head())


# --------------------------------------------------
# FINAL MAIN ROUTER
# --------------------------------------------------

def main():
    st.title("Financial Inclusion – UPI Adoption Prototype")

    file = st.sidebar.file_uploader("Upload dataset", ["csv","xlsx"])

    if file:
        sheets = load_sheets(file)
        df = merge_all_sheets_columnwise(sheets)
        df = build_upi_adoption_score(df)
        st.session_state["df"]=df

    df = st.session_state.get("df")
    if df is None:
        st.title("Upload your capstone dataset")
        return

    nav = st.sidebar.radio("Navigate", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])

    if nav=="Overview": page_overview(df)
    elif nav=="ML Model": page_ml(df)
    elif nav=="Time Series": page_ts(df)
    elif nav=="Text Analytics": page_text(df)
    elif nav=="Geo Dashboard": page_geo(df)


if __name__ == "__main__":
    main()
