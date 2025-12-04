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

# New plotting helper
import plotly.express as px
from textblob import TextBlob
import re

st.set_page_config(layout="wide", page_title="Financial Tracking Prototype")

# --------------------------------------------------
# LOAD ALL SHEETS
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
# MERGE ALL SHEETS COLUMN-WISE → OUTER JOIN → NO KEY
# --------------------------------------------------

def merge_all_sheets_columnwise(sheets):
    combined = None
    for name, df in sheets.items():
        df = df.copy().reset_index(drop=True)
        if combined is None:
            combined = df
        else:
            L = max(len(combined), len(df))
            combined = combined.reindex(range(L)).reset_index(drop=True)
            df = df.reindex(range(L)).reset_index(drop=True)
            # avoid collisions
            df.columns = [c if c not in combined.columns else f"{c}_{name}" for c in df.columns]
            combined = pd.concat([combined, df], axis=1, join="outer")

    combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    return combined


# --------------------------------------------------
# BUILD VALID DATE STRUCTURE FOR TS PAGE SAFELY
# --------------------------------------------------

def fix_dates(df):
    # Remove accidental duplicate date part columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Combine Year+Month manually if exists
    y = [c for c in df.columns if "year" in c.lower()]
    m = [c for c in df.columns if "month" in c.lower()]

    if y and m and "ts_date" not in df.columns:
        try:
            df["ts_date"] = pd.to_datetime(
                df[y[0]].astype(int).astype(str) + "-" +
                df[m[0]].astype(int).astype(str) + "-01",
                errors="coerce"
            )
        except:
            df["ts_date"] = pd.NaT
    return df


# --------------------------------------------------
# CLEAN DATA: FIX NUMERIC COLUMNS & DROP TEXT NUMERIC
# --------------------------------------------------

def clean_data(df):
    df = fix_dates(df)

    # Force all numeric-looking columns to actual numbers
    for c in df.columns:
        # detect numeric-ish columns stored as text
        text_sample = df[c].dropna().astype(str).head(20)
        numeric_like = text_sample.apply(lambda x: bool(re.fullmatch(r"[0-9.\-]+", x))).mean() > 0.7
        if numeric_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop columns that are still object but contain too many unique huge text entries
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        if df[c].nunique() > 300:  # high-unique text = bad feature
            df.drop(columns=[c], inplace=True)

    # Final fill NA for numbers
    for c in df.select_dtypes(include=np.number).columns:
        df[c].fillna(df[c].median(), inplace=True)

    return df


# --------------------------------------------------
# CREATE TARGET COLUMN USING ALL FACTORS → PCA
# --------------------------------------------------

def build_upi_adoption_score(df):
    df = clean_data(df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        df["UPI_Adoption_SScore"] = 50.0
        return df

    numeric_cols = [c for c in numeric_cols if c != "UPI_Adoption_SScore"]

    if not numeric_cols:
        df["UPI_Adoption_SScore"] = 50.0
        return df

    clean = df[numeric_cols]
    X = StandardScaler().fit_transform(clean)

    comp = PCA(1, random_state=42).fit_transform(X).ravel()
    score = 50.0 if comp.max()==comp.min() else (comp-comp.min())/(comp.max()-comp.min())*100

    df["UPI_Adoption_SScore"] = score
    return df


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

def page_overview(df):
    st.title("Financial Tracking Prototype")
    st.subheader("Dataset Overview")
    st.write(f"Total rows: {len(df):,}  |  Total columns: {len(df.columns):,}")
    st.dataframe(df.head())


# --------------------------------------------------
# ML MODEL PAGE (TRAIN + STATE WISE PREDICTION)
# --------------------------------------------------

def page_ml(df):
    st.subheader("Machine Learning – Adoption Score Prediction")

    target = "UPI_Adoption_SScore"
    if target not in df.columns:
        st.error("❌ Target score column missing.")
        return

    # Drop target safely
    X = df.drop(columns=[target], errors="ignore").copy()
    y = df[target]

    # Remove mixed-type object columns and normalize datetime in text
    for col in X.columns:
        if X[col].dtype == "object":
            # Convert datetime objects to string
            if X[col].apply(lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).any():
                X[col] = X[col].astype(str)

    # Select clean numeric and categorical columns
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Ensure categorical columns are now uniform
    for col in cat_cols:
        try:
            X[col] = X[col].astype(str)
        except:
            X.drop(columns=[col], inplace=True)

    # Preprocessing
    transformer = make_column_transformer(
        (SimpleImputer(strategy="median"), num_cols),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols),
        remainder="drop"
    )

    model = RandomForestRegressor(n_estimators=700, random_state=42, n_jobs=-1)
    pipe = make_pipeline(transformer, model)

    # Train-test split
    test_split = st.sidebar.slider("Test split (%)", 10, 40, 20) / 100
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split, random_state=42)

    # Train model on safe input
    if st.button("Train ML Model"):
        try:
            pipe.fit(X_tr, y_tr)
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            return

        # Predict & evaluate
        pr = pipe.predict(X_te)
        r2 = r2_score(y_te, pr)
        mae = mean_absolute_error(y_te, pr)
        rmse = mean_squared_error(y_te, pr)**0.5

        st.metric("R² Accuracy", f"{r2 * 100:.2f}%")
        st.metric("MAE", round(mae, 4))
        st.metric("RMSE", round(rmse, 4))

        # Plot
        fig = px.scatter(pd.DataFrame({"Actual": y_te, "Predicted": pr}),
                         x="Actual", y="Predicted", trendline="ols",
                         title="Actual vs Predicted Adoption Score")
        st.plotly_chart(fig, use_container_width=True)

        # State-wise prediction
        if "State" in df.columns:
            st.write("### State-wise Adoption Score Prediction")
            geo = df.groupby("State", as_index=False)["UPI_Adoption_SScore"].mean()
            X_full = X.copy()
            try:
                geo["Predicted_Score"] = pipe.predict(X_full)
            except:
                geo["Predicted_Score"] = np.nan
            st.dataframe(geo)


# --------------------------------------------------
# TIME SERIES PAGE
# --------------------------------------------------

def page_ts(df):
    st.subheader("Time Series Forecasting – Digital Transaction Volume")

    # Detect safe date column
    date_cols = [c for c in df.columns if c.lower() in ["ts_date","date","transaction_date","order_date"]]
    nums = df.select_dtypes(include=np.number).columns.tolist()

    if not nums:
        st.error("❌ No numeric columns found for forecasting!")
        return

    vol_col = st.selectbox("Select transaction volume column", nums)

    if date_cols:
        date_col = st.selectbox("Select valid date column", date_cols)
        ts = df[[date_col, vol_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    else:
        st.error("❌ No valid date column detected for time series!")
        return

    ts = ts.dropna(subset=[date_col,vol_col]).sort_values(date_col)
    if ts.empty:
        st.error("❌ No valid date rows remain after conversion!")
        return

    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(350, random_state=42, n_jobs=-1).fit(ts[["t"]], ts[vol_col].values)
    h = st.slider("Forecast months", 3, 36, 12)
    future_dates = pd.date_range(start=ts[date_col].iloc[-1], periods=h+1, freq="M")[1:]
    future_t = np.arange(ts["t"].iloc[-1]+1, ts["t"].iloc[-1]+1+h)
    fut_preds = model.predict(future_t.reshape(-1,1))

    fut_df = pd.DataFrame({date_col:future_dates,vol:fut_preds,"Type":"Forecast"})
    hist_df = ts[[date_col,vol]].copy(); hist_df["Type"]="Actual"

    st.plotly_chart(px.line(pd.concat([hist_df,fut_df],ignore_index=True),
                            x=date_col,y=vol,color="Type",title="Actual vs Forecast"),
                            use_container_width=True)
    st.dataframe(fut_df.head())


# --------------------------------------------------
# TEXT PAGE
# --------------------------------------------------

def page_text(df):
    st.subheader("Text Analysis – Topics + Sentiment")

    txts = df.select_dtypes(include="object").columns.tolist()

    if not txts:
        st.error("❌ No text columns found for NLP")
        return

    tcol = st.selectbox("Select text column", txts)
    data = df[tcol].dropna().astype(str)

    if data.empty:
        st.warning("⚠ No valid text rows left")
        return

    k = st.slider("Number of topics",2,6,3)
    vec = TfidfVectorizer(1500, stop_words="english")
    X = vec.fit_transform(data)
    nmf = NMF(k, random_state=42, init="nndsvda").fit(X)
    words = vec.get_feature_names_out()

    for i,t in enumerate(nmf.components_):
        topw = [words[idx] for idx in t.argsort()[-10:][::-1]]
        st.write(f"**Topic {i+1}:** "+", ".join(topw))

    if st.checkbox("Run sentiment analysis"):
        st.dataframe(data.apply(lambda x: TextBlob(x).sentiment.polarity).head())


# --------------------------------------------------
# GEO MAP PAGE → ONLY STATE COLUMN
# --------------------------------------------------

def page_geo(df):
    st.subheader("Geo Dashboard – Interactive State View")

    if "State" not in df.columns:
        st.error("❌ `State` column not found!")
        return

    geo = df.groupby("State", as_index=False)["UPI_Adoption_SScore"].mean()

    fig = px.choropleth(
        geo,
        locations="State",
        locationmode="country names",
        color="UPI_Adoption_SScore",
        title="State-wise UPI Adoption Score (India)",
        geojson=None,
        scope="asia"
    )

    st.write("### Click on a State or select below")
    st.plotly_chart(fig, use_container_width=True)

    chosen = st.selectbox("Select State", ["(none)"] + list(geo["State"]))
    if chosen != "(none)":
        score = geo.loc[geo["State"]==chosen, "UPI_Adoption_SScore"].values[0]
        st.success(f"📍 {chosen} selected!")
        st.metric("Adoption Score", round(score,4))

    st.write("### Adoption Score by State")
    st.dataframe(geo)


# --------------------------------------------------
# APP ROUTER
# --------------------------------------------------

def main():
    st.sidebar.header("Upload Capstone Dataset")
    file = st.sidebar.file_uploader("Upload file", ["csv","xlsx"])

    if "plotly_click" not in st.session_state:
        st.session_state["plotly_click"] = None

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

