import streamlit as st
import pandas as p
import numpy as n

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
import plotly.express as x
import pydeck as d


st.set_page_config(layout="wide", page_title="UPI Adoption Tracker")

# ---- Load mult-sheet Excel or CSV ----
@st.cache_data
def load_sheets(file):
    if file.name.lower().endswith(".csv"):
        df = p.read_csv(file)
        df.columns = df.columns.str.strip().str.title()
        return {"Main": df}
    else:
        xl = p.ExcelFile(file)
        sh = {s: xl.parse(s) for s in xl.sheet_names}
        for k in sh:
            sh[k].columns = sh[k].columns.str.strip().str.title()
        return sh

# ---- Merge sheets column-wise with outer join on index ----
def merge_all_sheets(sheets):
    df = None
    for name, dff in sheets.items():
        dff = dff.copy().reset_index(drop=True)
        if df is None:
            df = dff
        else:
            L = max(len(df), len(dff))
            df = df.reindex(range(L))
            dff = dff.reindex(range(L))
            # Avoid duplicate column names
            dff.columns = [c if c not in df.columns else f"{c}_{name}" for c in dff.columns]
            df = p.concat([df, dff], axis=1)
    return df

# ---- Create synthetic UPI Adoption Score via PCA ----
def compute_upi_score(df):
    numc = df.select_dtypes(include=n.number).columns.tolist()
    if not numc:
        df["UPI_Adoption_Score"] = 50.0
        return df
    X = StandardScaler().fit_transform(df[numc].fillna(df[numc].median()))
    pc = PCA(1, random_state=42).fit_transform(X).ravel()
    sc = 50.0 if pc.max()==pc.min() else (pc-pc.min())/(pc.max()-pc.min())*100
    df["UPI_Adoption_Score"] = sc
    return df

# ---- Overview Page ----
def page_overview(df):
    st.subheader("Combined Dataset Overview")
    st.write(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]:,}")
    st.dataframe(df.head())

# ---- ML Model Page ----
def page_ml(df):
    st.subheader("ML: Predict UPI Adoption Score")
    t = "UPI_Adoption_Score"
    X = df.drop(columns=[t], errors="ignore")
    y = df[t]

    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(include=n.number).columns.tolist()

    tr = ColumnTransformer([
        ("num", Pipeline([("i",SimpleImputer("median")),("s",StandardScaler())]), num),
        ("cat", Pipeline([("i",SimpleImputer("most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ], remainder="drop")

    m = RandomForestRegressor(500,random_state=42,n_jobs=-1)
    pipe = Pipeline([("p", tr), ("m", m)])

    sp = st.sidebar.slider("Test split",0.1,0.4,0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=sp,random_state=42)
    if st.button("Train ML Model"):
        pipe.fit(X_tr,y_tr)
        pr = pipe.predict(X_te)
        st.metric("R²",f"{r2_score(y_te,pr):.4f}")
        st.metric("MAE",f"{mean_absolute_error(y_te,pr):.4f}")
        st.metric("RMSE",f"{mean_squared_error(y_te,pr)**0.5:.4f}")
        st.plotly_chart(x.scatter(p.DataFrame({"Actual":y_te,"Pred":pr}),x="Actual",y="Pred",trendline="ols"),use_container_width=True)

# ---- Time Series Page ----
def page_ts(df):
    st.subheader("Time Series Forecast")
    D = [c for c in df.columns if "date" in c.lower()]
    Y = [c for c in df.columns if "year" in c.lower()]
    M = [c for c in df.columns if "month" in c.lower()]
    nums = df.select_dtypes(include=n.number).columns.tolist()

    vol = st.selectbox("Transaction Volume Column", nums)

    if D:
        dat = st.selectbox("Date Column", D)
        ts = df[[dat, vol]].copy()
        ts[dat] = p.to_datetime(ts[dat], errors="coerce")
        dc = dat
    elif Y and M:
        ycol = st.selectbox("Year Column", Y)
        mcol = st.selectbox("Month Column", M)
        ts = df[[ycol, mcol, vol]].copy()
        for c in [ycol,mcol,vol]:
            ts[c] = p.to_numeric(ts[c], errors="coerce")
        ts = ts.dropna(subset=[ycol,mcol,vol])
        if ts.empty:
            st.error("No valid year/month rows")
            return
        ts["ts_date"] = p.to_datetime(
            ts[ycol].astype(int).astype(str)+"-"+ts[mcol].astype(int).astype(str)+"-01",
            errors="coerce"
        )
        dc = "ts_date"
    else:
        st.error("No valid date columns")
        return

    ts = ts.dropna(subset=[dc, vol]).sort_values(dc)
    if ts.empty:
        st.error("No valid rows after date cleaning")
        return

    ts["t"] = n.arange(len(ts))
    model = RandomForestRegressor(300,random_state=42,n_jobs=-1).fit(ts[["t"]], ts[vol].values)

    h = st.slider("Forecast months",3,36,12)
    last = ts[dc].iloc[-1]
    f_date = p.date_range(start=last,periods=h+1,freq="M")[1:]
    f_t = n.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+h)
    f_pr = model.predict(f_t.reshape(-1,1))

    fdf = p.DataFrame({dc:f_date,vol:f_pr,"type":"Forecast"})
    hist = ts[[dc,vol]].copy(); hist["type"]="Actual"
    st.plotly_chart(x.line(p.concat([hist,fdf],ignore_index=True),x=dc,y=vol,color="type"),use_container_width=True)
    st.dataframe(fdf.head())

# ---- Text Analytics Page ----
def page_text(df):
    st.subheader("Text Analytics")
    txts = df.select_dtypes(include="object").columns.tolist()
    if not txts:
        st.warning("No text columns")
        return
    col = st.selectbox("Text Column", txts)
    dta = df[col].dropna().astype(str)
    k = st.slider("Topics",2,6,3)
    v = TfidfVectorizer(max_features=1500,stop_words="english").fit_transform(dta)
    nmf = NMF(k,random_state=42,init="nndsvda").fit(v)
    w = TfidfVectorizer().fit(dta).get_feature_names_out()
    for i,t in enumerate(nmf.components_): 
        st.write(f"Topic {i+1}: " + ", ".join([w[idx] for idx in t.argsort()[-10:][::-1]]))
    if st.checkbox("Sentiment"):
        st.dataframe(dta.apply(lambda x: TextBlob(x).sentiment.polarity).head())

# ---- Geo Dashboard Page ----
def page_geo(df):
    st.subheader("Geo Dashboard")

    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if not lat_cols or not lon_cols:
        st.error("No coordinates for geo")
        return

    lab = st.selectbox("District/Label Column", df.columns)
    lat = st.selectbox("Latitude Column", lat_cols)
    lon = st.selectbox("Longitude Column", lon_cols)

    geo = df[[lab, lat, lon, "UPI_Adoption_Score"]].copy()
    for c in [lat,lon,"UPI_Adoption_Score"]:
        geo[c] = p.to_numeric(geo[c], errors="coerce")

    geo = geo.dropna(subset=[lat,lon,"UPI_Adoption_Score"])
    if geo.empty:
        st.error("No valid geo rows after numeric cleaning")
        return

    r = geo["UPI_Adoption_Score"]
    geo["radius"] = 2000 + (r - r.min())/(r.max()-r.min()+1e-6)*8000

    layer = d.Layer("ScatterplotLayer",geo,get_position=[lon,lat],get_radius="radius",pickable=True)
    st.pydeck_chart(d.Deck(layers=[layer],initial_view_state=d.ViewState(latitude=geo[lat].mean(),longitude=geo[lon].mean(),zoom=4),tooltip={"text":f"{lab}\nUPI Adoption: {{UPI_Adoption_Score}}"}))
    st.dataframe(geo.head())

# ---- Router ----
def main():
    if "df" not in st.session_state:
        st.session_state["df"] = None

    file = st.sidebar.file_uploader("Upload dataset", ["csv","xlsx"])
    if file:
        sheets = load_sheets(file)
        df = merge_all_sheets(sheets)
        df = build_upi_adoption_score(df)
        st.session_state["df"] = df

    if st.session_state["df"] is None:
        st.title("Upload your dataset to begin")
        return

    df = st.session_state["df"]

    nav = st.sidebar.radio("Navigate", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])
    if nav=="Overview": page_overview(df)
    elif nav=="ML Model": page_ml(df)
    elif nav=="Time Series": page_ts(df)
    elif nav=="Text Analytics": page_text(df)
    elif nav=="Geo Dashboard": page_geo(df)

if __name__ == "__main__":
    main()
