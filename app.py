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
import plotly.express as px
from textblob import TextBlob
import pydeck as pdk  # retained but not using for states map


st.set_page_config(page_title="UPI Adoption Prototype", layout="wide")


# --------------------------------------------------
# 1. DATA LOADING & COLUMN-WISE OUTER MERGE
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
    df = None
    for name, dff in sheets.items():
        dff = dff.copy().reset_index(drop=True)
        if df is None:
            df = dff
        else:
            L = max(len(df), len(dff))
            df = df.reindex(range(L)).reset_index(drop=True)
            dff = dff.reindex(range(L)).reset_index(drop=True)
            # prevent collisions
            dff.columns = [c if c not in df.columns else f"{c}_{name}" for c in dff.columns]
            df = pd.concat([df, dff], axis=1, join="outer")

    df = df.loc[:, ~df.columns.duplicated(keep='first')]  # drop accidental duplicates
    return df


def build_upi_adoption_score(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        df["UPI_Adoption_Score"] = 50.0
        return df, []

    clean = df[numeric_cols].fillna(df[numeric_cols].median())
    X = StandardScaler().fit_transform(clean)

    comp = PCA(1, random_state=42).fit_transform(X).ravel()
    score = 50.0 if comp.max()==comp.min() else (comp-comp.min())/(comp.max()-comp.min())*100

    df["UPI_Adoption_Score"] = score
    return df, numeric_cols


# --------------------------------------------------
# 2. PAGES
# --------------------------------------------------

def page_overview(df):
    st.subheader("Dataset Preview")
    st.write(f"Total rows: {df.shape[0]:,}  |  Total columns: {df.shape[1]:,}")
    st.dataframe(df.head())


def page_ml(df):
    st.subheader("Machine Learning Model – Predict Adoption Score")

    target = "UPI_Adoption_Score"
    if target not in df.columns:
        st.error("Target score column missing.")
        return

    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    transformer = make_column_transformer(
        (SimpleImputer(strategy="median"), num_cols),
        (SimpleImputer(strategy="most_frequent"), cat_cols),
        remainder="drop"
    )

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    pipe = make_pipeline(transformer, model)

    split = st.sidebar.slider("Test split %", 0.1, 0.4, 0.2)
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

        fig = px.scatter(pd.DataFrame({"Actual":y_te, "Predicted":pr}), x="Actual", y="Predicted", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)


def page_ts(df):
    st.subheader("Time Series Forecast – Digital Transactions")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found for time series.")
        return

    vol = "UPI_Transaction_Volume" if "UPI_Transaction_Volume" in df.columns else numeric_cols[0]
    vol_col = st.selectbox("Select volume column", numeric_cols, index=numeric_cols.index(vol) if vol in numeric_cols else 0)

    ts = None
    date_col = None

    if date_cols:
        dcol = st.selectbox("Select date column", date_cols)
        ts = df[[dcol, vol_col]].copy()
        ts[dcol] = pd.to_datetime(ts[dcol], errors="coerce")
        date_col = dcol

    elif year_cols and month_cols:
        y = st.selectbox("Select year column", year_cols)
        m = st.selectbox("Select month column", month_cols)
        ts = df[[y, m, vol_col]].copy()
        for c in [y, m, vol_col]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")
        ts = ts.dropna(subset=[y, m, vol_col])
        if ts.empty:
            st.error("No valid year/month rows left.")
            return
        ts["ts_date"] = pd.to_datetime(ts[y].astype(int).astype(str)+"-"+ts[m].astype(int).astype(str)+"-01", errors="coerce")
        date_col = "ts_date"

    if ts is None or date_col is None:
        st.error("No usable date structure found.")
        return

    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)
    if ts.empty:
        st.error("No valid rows after date cleaning.")
        return

    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(ts[["t"]], ts[vol_col].values)

    steps = st.slider("Forecast months", 3, 24, 12)
    last_date = ts[date_col].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=steps+1, freq="M")[1:]
    future_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+steps)
    future_preds = model.predict(future_t.reshape(-1,1))

    fut_df = pd.DataFrame({date_col:future_dates, vol_col:future_preds, "Type":"Forecast"})
    hist_df = ts[[date_col, vol_col]].copy()
    hist_df["Type"]="Actual"

    fig = px.line(pd.concat([hist_df, fut_df.rename(columns={"Type":"Type"})], ignore_index=True),
                  x=date_col,y=vol_col,color="Type")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fut_df.head())


def page_text(df):
    st.subheader("Text Analysis – NMF Topics + Sentiment")

    txts = df.select_dtypes(include="object").columns.tolist()
    if not txts:
        st.warning("No text columns found.")
        return

    tcol = st.selectbox("Select text column", txts)
    data = df[tcol].dropna().astype(str)

    if data.empty:
        st.warning("No text rows available after dropna().")
        return

    k = st.slider("Number of topics", 2, 6, 3)
    vec = TfidfVectorizer(max_features=1500, stop_words="english")
    X = vec.fit_transform(data)
    nmf = NMF(n_components=k, random_state=42, init="nndsvda").fit(X)
    words = vec.get_feature_names_out()

    for i, t in enumerate(nmf.components_):
        topw = [words[idx] for idx in t.argsort()[-10:][::-1]]
        st.write(f"Topic {i+1}: " + ", ".join(topw))

    if st.checkbox("Run sentiment analysis"):
        scores = data.apply(lambda t: TextBlob(t).sentiment.polarity)
        st.plotly_chart(px.histogram(scores, nbins=25, title="Sentiment Distribution"), use_container_width=True)
        st.dataframe(scores.head())


def page_geo(df):
    st.subheader("Interactive Geo Dashboard – India States")

    if "State" not in df.columns:
        st.error("No `State` column found!")
        return

    geo = df.groupby("State", as_index=False)["UPI_Adoption_SScore"].mean()
    geo["Country"] = "India"

    fig = px.choropleth(
        geo,
        locations="Country",
        hover_name="State",
        color="UPI_Adoption_SScore",
        scope="asia",
        locationmode="country names",
        title="UPI Adoption Score Across India States",
    )

    # Manual state selection
    clicked = st.plotly_chart(fig, use_container_width=True)

    st.write("### Or select a State from dropdown:")
    chosen = st.selectbox("Select State:", ["(none)"] + list(geo["State"]))
    if chosen != "(none)":
        score = geo.loc[geo["State"] == chosen, "UPI_Adoption_SScore"].values[0]
        st.success(f"📍 {chosen} Selected!")
        st.metric("UPI Adoption Score:", round(score, 4))

    st.dataframe(geo)


# --------------------------------------------------
# 3. APP ROUTER
# --------------------------------------------------

def main():
    st.sidebar.header("Upload Capstone Dataset")
    file = st.sidebar.file_uploader("Upload file", ["csv","xlsx"])

    if file:
        sheets = load_sheets(file)
        df = merge_all_sheets_columnwise(sheets)
        df, _ = build_upi_adoption_score(df)
        st.session_state["df"] = df

    df = st.session_state.get("df")
    if df is None:
        st.title("Upload your dataset to begin")
        return

    nav = st.sidebar.radio("Navigate", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])
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
