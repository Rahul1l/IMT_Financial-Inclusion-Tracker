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

st.set_page_config(page_title="UPI Adoption Prototype", layout="wide")

# --------------------------------------------------
# DATA LOADING & MERGING
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
    """Outer merge all sheets purely column-wise by index"""
    combined = None
    for name, df in sheets.items():
        df = df.copy().reset_index(drop=True)
        if combined is None:
            combined = df
        else:
            L = max(len(combined), len(df))
            combined = combined.reindex(range(L)).reset_index(drop=True)
            df = df.reindex(range(L)).reset_index(drop=True)
            # Avoid column name collisions
            df.columns = [c if c not in combined.columns else f"{c}_{name}" for c in df.columns]
            combined = pd.concat([combined, df], axis=1, join="outer")

    combined = combined.loc[:, ~combined.columns.duplicated()]  # Drop accidental duplicates
    return combined


def build_upi_adoption_score(df):
    """Create synthetic adoption score from all numeric factor sheets using PCA"""
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
        df["UPI_Adoption_SScore"] = 50.0
        return df

    clean = df[numeric].fillna(df[numeric].median())
    X = StandardScaler().fit_transform(clean)

    comp = PCA(1, random_state=42).fit_transform(X).ravel()
    score = 50.0 if comp.max()==comp.min() else (comp - comp.min())/(comp.max() - comp.min()) * 100

    df["UPI_Adoption_SScore"] = score
    return df


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

def page_overview(df):
    st.title("UPI Adoption – Financial Inclusion Prototype")
    st.subheader("Dataset Overview")
    st.write(f"Total rows: {df.shape[0]:,}  |  Total columns: {df.shape[1]:,}")
    st.dataframe(df.head())

    st.write("### State-wise Adoption Score Preview")
    if "State" in df.columns:
        state_preview = df.groupby("State", as_index=False)["UPI_Adoption_SScore"].mean()
        st.dataframe(state_preview.head())
    else:
        st.write("`State` column not found – state grouping skipped on overview")


# --------------------------------------------------
# ML MODEL PAGE
# --------------------------------------------------

def page_ml(df):
    st.subheader("Machine Learning – Predict Adoption Score")

    target = "UPI_Adoption_SScore"

    if target not in df.columns:
        st.error(f"❌ Required target `{target}` is missing.")
        return

    # Features/Target
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Preprocessing
    transformer = make_column_transformer(
        (SimpleImputer(strategy="median"), num_cols),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols),
        remainder="drop"
    )

    model = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    pipe = make_pipeline(transformer, model)

    # Split control
    split = st.sidebar.slider("Test split (%)", 10, 40, 20) / 100
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split, random_state=42)

    # Train
    if st.button("Train ML Model"):
        pipe.fit(X_tr, y_tr)
        pr = pipe.predict(X_te)

        r2 = r2_score(y_te, pr)
        mae = mean_absolute_error(y_te, pr)
        rmse = mean_squared_error(y_te, pr)**0.5

        st.metric("R² Accuracy", f"{r2*100:.2f}%")
        st.metric("MAE", round(mae, 4))
        st.metric("RMSE", round(rmse, 4))

        st.plotly_chart(px.scatter(pd.DataFrame({"Actual":y_te, "Predicted":pr}), x="Actual",y="Predicted", trendline="ols"),use_container_width=True)


# --------------------------------------------------
# TIME SERIES FORECAST PAGE
# --------------------------------------------------

def page_ts(df):
    st.subheader("Time Series Forecasting – Digital Transaction Volume")

    # Detect possible date structures
    date_cols = [c for c in df.columns if "date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns found for forecasting!")
        return

    # Select volume column
    default_volume = "Upi Transaction Volume" if "Upi Transaction Volume" in df.columns else numeric_cols[0]
    vol_col = st.selectbox("Select transaction volume column", numeric_cols,
                           index=numeric_cols.index(default_volume) if default_volume in numeric_cols else 0)

    ts = None
    date_col = None

    # Case A: Which contains a full date column
    if date_cols:
        date_col = st.selectbox("Select date column", date_cols)
        ts = df[[date_col, vol_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")

    # Case B: Which contains separate year/month
    elif year_cols and month_cols:
        y = st.selectbox("Select year", year_cols)
        m = st.selectbox("Select month", month_cols)
        ts = df[[y, m, vol_col]].copy()
        for c in [y,m,vol_col]:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")
        ts = ts.dropna(subset=[y,m,vol_col])
        ts["ts_date"] = pd.to_datetime(
            ts[y].astype(int).astype(str) + "-" + ts[m].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        date_col = "ts_date"

    else:
        st.error("❌ No date or year/month structure found for forecasting!")
        return

    # Final cleaning and sorting
    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)

    if ts.empty:
        st.error("❌ No valid rows left after date conversion!")
        return

    # Train trend learner
    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(300, random_state=42, n_jobs=-1)
    model.fit(ts[["t"]], ts[vol_col].values)

    # Forecasting input
    steps = st.slider("Forecast months", 3, 24, 12)
    last_date = ts[date_col].iloc[-1]
    f_dates = pd.date_range(start=last_date, periods=steps+1, freq="M")[1:]
    f_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+steps)
    f_pr = model.predict(f_t.reshape(-1,1))

    f_df = pd.DataFrame({date_col:f_dates, vol_col:f_pr, "Type":"Forecast"})
    hist_df = ts[[date_col, vol_col]].copy(); hist_df["Type"]="Actual"

    # Plot
    st.plotly_chart(px.line(pd.concat([hist_df,f_df.rename(columns={"Type":"Type"})],ignore_index=True),
                            x=date_col,y=vol_col,color="Type"),use_container_width=True)
    st.dataframe(f_df.head())


# --------------------------------------------------
# TEXT ANALYTICS PAGE
# --------------------------------------------------

def page_text(df):
    st.subheader("Text Analytics – Topics & Sentiment")

    txts = df.select_dtypes(include="object").columns.tolist()
    if not txts:
        st.warning("⚠ No text columns found for NLP")
        return

    tcol = st.selectbox("Select text column for topic modeling", txts)
    data = df[tcol].dropna().astype(str)

    if data.empty:
        st.warning("⚠ No text rows after cleaning")
        return

    # Topic modeling
    k = st.slider("Number of topics", 2, 6, 3)
    vec = TfidfVectorizer(max_features=1500, stop_words="english")
    X = vec.fit_transform(data)
    nmf = NMF(n_components=k,random_state=42,init="nndsvda").fit(X)
    words = vec.get_feature_names_out()

    for i, t in enumerate(nmf.components_):
        topw = [words[idx] for idx in t.argsort()[-10:][::-1]]
        st.write(f"**Topic {i+1}:** " + ", ".join(topw))

    # Sentiment
    if st.checkbox("Run sentiment analysis"):
        sc = data.apply(lambda x: TextBlob(x).sentiment.polarity)
        st.plotly_chart(px.histogram(sc, nbins=25, title="Sentiment Distribution"),use_container_width=True)
        st.dataframe(sc.head())


# --------------------------------------------------
# GEO DASHBOARD PAGE (INDIA STATES SELECTION ENABLED)
# --------------------------------------------------

def page_geo(df):
    st.subheader("Geo Dashboard – UPI Adoption Score by State")

    if "State" not in df.columns:
        st.error("❌ `State` column not found!")
        return

    # Aggregate Adoption Score by State
    geo = df.groupby("State", as_index=False)["UPI_Adoption_SScore"].mean()

    # Create interactive India map
    fig = px.choropleth(
        geo,
        locations="State",
        locationmode="country names",
        color="UPI_Adoption_SScore",
        title="UPI Adoption Score Across India States",
        scope="asia"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write("### Select a state to view score:")

    # Dropdown for selection
    chosen = st.selectbox("Select State", ["(none)"] + list(geo["State"]))
    if chosen != "(none)":
        score = geo.loc[geo["State"]==chosen, "UPI_Adoption_SScore"].values[0]
        st.success(f"📍 {chosen} selected!")
        st.metric("Adoption Score", round(score,4))

    st.write("### Adoption Score by State Table")
    st.dataframe(geo)


# --------------------------------------------------
# MAIN APP ROUTER
# --------------------------------------------------

def main():
    file = st.sidebar.file_uploader("Upload dataset", ["csv","xlsx"])
    if file:
        sheets = load_sheets(file)
        df = merge_all_sheets_columnwise(sheets)
        df = build_upi_adoption_score(df)
        st.session_state["df"] = df

    df = st.session_state.get("df")
    if df is None:
        st.title("Upload your dataset to begin")
        return

    nav = st.sidebar.radio("Navigate", ["Overview","ML Model","Time Series","Text Analytics","Geo Dashboard"])

    if nav=="Overview": page_overview(df)
    elif nav=="ML Model": page_ml(df)
    elif nav=="Time Series": page_ts(df)
    elif nav=="Text Analytics": page_text(df)
    elif nav=="Geo Dashboard": page_geo(df)


if __name__ == "__main__":
    main()
