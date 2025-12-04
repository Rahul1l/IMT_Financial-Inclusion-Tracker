import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from textblob import TextBlob
import plotly.express as px

st.set_page_config(page_title="UPI Adoption Analytics", layout="wide")

# --------------------------------------------------
# 1. LOAD ALL SHEETS & MERGE COLUMN-WISE
# --------------------------------------------------

@st.cache_data
def load_sheets(file):
    """Load CSV or Excel with multiple sheets."""
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


def merge_all_sheets_columnwise(sheets: dict) -> pd.DataFrame:
    """
    Simply merge ALL sheets column-wise (outer join on index).
    No dropping or cleaning – every column from all sheets is kept.
    """
    combined = None
    for name, df in sheets.items():
        df = df.copy().reset_index(drop=True)

        if combined is None:
            combined = df
        else:
            max_len = max(len(combined), len(df))
            combined = combined.reindex(range(max_len)).reset_index(drop=True)
            df = df.reindex(range(max_len)).reset_index(drop=True)

            # Avoid duplicate column names
            new_cols = []
            existing = set(combined.columns)
            for c in df.columns:
                if c in existing or c in new_cols:
                    new_cols.append(f"{c}_{name}")
                else:
                    new_cols.append(c)
            df.columns = new_cols

            combined = pd.concat([combined, df], axis=1)

    # Keep only first occurrence of exact duplicate column names (if any remain)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    return combined


# --------------------------------------------------
# 2. BUILD SYNTHETIC ADOPTION SCORE (NUMERIC ONLY)
# --------------------------------------------------

def build_upi_adoption_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create UPI_Adoption_Score from all numeric columns using PCA.
    Text columns (like Report_Text) are untouched and kept.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        df["UPI_Adoption_Score"] = 50.0
        return df

    X = df[num_cols].copy()
    X = X.fillna(X.median())

    # PCA 1 component
    pca = PCA(n_components=1, random_state=42)
    comp = pca.fit_transform(X).ravel()

    if comp.max() == comp.min():
        score = np.full_like(comp, 50.0, dtype=float)
    else:
        score = (comp - comp.min()) / (comp.max() - comp.min()) * 100.0

    df["UPI_Adoption_Score"] = score
    return df


# --------------------------------------------------
# 3. PAGES
# --------------------------------------------------

def page_overview(df: pd.DataFrame):
    st.subheader("Combined Dataset Overview (All Sheets)")
    st.write(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]:,}")
    st.dataframe(df.head())

    with st.expander("Column summary"):
        info = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str).values,
            "Non-null": df.notnull().sum().values,
            "Nulls": df.isnull().sum().values,
        })
        st.dataframe(info)


def page_ml(df: pd.DataFrame):
    st.subheader("ML Model – Predict Synthetic UPI Adoption Score")

    target = "UPI_Adoption_SScore"
    if target not in df.columns:
        st.error("❌ UPI_Adoption_SScore column missing. Re-upload data.")
        return

    # Use numeric features only to avoid encoder issues
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    if not num_cols:
        st.error("❌ No numeric feature columns found for ML training.")
        return

    X = df[num_cols].fillna(df[num_cols].median())
    y = df[target]

    mask = y.notna()
    X = X[mask]
    y = y[mask]

    test_size = st.sidebar.slider("Test split (%)", 10, 40, 20) / 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)**0.5 if False else mean_squared_error(y_test, preds)**0.5

        st.metric("R²", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

        comp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
        fig = px.scatter(
            comp_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Adoption Score"
        )
        st.plotly_chart(fig, use_container_width=True)



def page_time_series(df: pd.DataFrame):
    st.subheader("Time Series Forecast – Digital Transaction Volume")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    year_cols = [c for c in df.columns if "year" in c.lower()]
    month_cols = [c for c in df.columns if "month" in c.lower()]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.error("No numeric columns found for forecasting.")
        return

    vol_col = st.selectbox("Select volume column", num_cols)

    ts = None
    date_col = None

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
            st.error("No valid Year/Month rows left after cleaning.")
            return

        ts["ts_date"] = pd.to_datetime(
            ts[y].astype(int).astype(str) + "-" +
            ts[m].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        date_col = "ts_date"
    else:
        st.error("No valid date or (year, month) structure found.")
        return

    ts = ts.dropna(subset=[date_col, vol_col]).sort_values(date_col)
    if ts.empty:
        st.error("No valid rows left after date cleaning.")
        return

    # Simple RF trend model on time index
    ts["t"] = np.arange(len(ts))
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(ts[["t"]], ts[vol_col])

    steps = st.slider("Forecast months", 3, 24, 12)
    last_date = ts[date_col].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=steps+1, freq="M")[1:]
    future_t = np.arange(ts["t"].iloc[-1] + 1, ts["t"].iloc[-1] + 1 + steps)
    future_preds = model.predict(future_t.reshape(-1, 1))

    fut_df = pd.DataFrame({
        date_col: future_dates,
        vol_col: future_preds,
        "Type": "Forecast"
    })

    hist_df = ts[[date_col, vol_col]].copy()
    hist_df["Type"] = "Actual"

    fig = px.line(pd.concat([hist_df, fut_df], ignore_index=True),
                  x=date_col, y=vol_col, color="Type",
                  title="Actual vs Forecast – Transaction Volume")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fut_df.head())


def page_text(df: pd.DataFrame):
    st.subheader("Text Analytics – Topics & Sentiment from Report_Text")

    if "Report_Text" not in df.columns:
        st.error("`Report_Text` column not found. Make sure your 7th sheet has this column name.")
        return

    texts = df["Report_Text"].dropna().astype(str)
    if texts.empty:
        st.warning("No non-empty text rows found in Report_Text.")
        return

    # Topic modeling
    n_topics = st.slider("Number of topics", 2, 6, 3)
    vectorizer = TfidfVectorizer(max_features=1500, stop_words="english")
    X = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda")
    nmf.fit(X)
    words = vectorizer.get_feature_names_out()

    st.markdown("### Key Themes")
    for i, topic in enumerate(nmf.components_):
        top_words = [words[idx] for idx in topic.argsort()[-10:][::-1]]
        st.write(f"**Topic {i+1}:** " + ", ".join(top_words))

    # Sentiment
    if st.checkbox("Run sentiment analysis"):
        sent = texts.apply(lambda t: TextBlob(t).sentiment.polarity)
        fig = px.histogram(sent, nbins=25, title="Sentiment Polarity Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sent.head())


def page_geo(df: pd.DataFrame):
    st.subheader("Geo Dashboard – UPI Adoption Score by State")

    if "State" not in df.columns:
        st.error("`State` column not found in combined dataset.")
        return

    if "UPI_Adoption_Score" not in df.columns:
        st.error("`UPI_Adoption_Score` not found – cannot draw map.")
        return

    geo = df.groupby("State", as_index=False)["UPI_Adoption_Score"].mean()

    fig = px.choropleth(
        geo,
        locations="State",
        locationmode="country names",
        color="UPI_Adoption_Score",
        title="UPI Adoption Score Across Indian States",
        scope="asia"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Inspect a State")
    state_choice = st.selectbox("Select State", ["(none)"] + list(geo["State"]))
    if state_choice != "(none)":
        val = geo.loc[geo["State"] == state_choice, "UPI_Adoption_Score"].values[0]
        st.success(f"📍 {state_choice} selected")
        st.metric("UPI Adoption Score", f"{val:.2f}")

    st.dataframe(geo)


# --------------------------------------------------
# 4. MAIN APP ROUTER
# --------------------------------------------------

def main():
    st.sidebar.header("Upload Dataset")
    file = st.sidebar.file_uploader("Upload Excel/CSV (with 7 sheets including text sheet)", ["csv", "xlsx"])

    if file:
        sheets = load_sheets(file)
        combined = merge_all_sheets_columnwise(sheets)
        combined = build_upi_adoption_score(combined)
        st.session_state["df"] = combined

    df = st.session_state.get("df")

    if df is None:
        st.title("Upload your capstone dataset to begin")
        return

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "ML Model", "Time Series", "Text Analytics", "Geo Dashboard"]
    )

    if page == "Overview":
        page_overview(df)
    elif page == "ML Model":
        page_ml(df)
    elif page == "Time Series":
        page_time_series(df)
    elif page == "Text Analytics":
        page_text(df)
    elif page == "Geo Dashboard":
        page_geo(df)


if __name__ == "__main__":
    main()

