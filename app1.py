# app.py — Final Streamlit FAANG Dashboard (StandardScaler + cleaned message)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from pathlib import Path
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="📈 FAANG Stock Predictor", layout="wide")


# --------------------------
# CONFIG — local artifact paths
# --------------------------
MODEL_PATH = r"C:\Users\Sree\OneDrive\Desktop\faang\best_model.pkl"
FEATURE_COLS_PATH = r"C:\Users\Sree\OneDrive\Desktop\faang\feature_columns.pkl"
SCALER_PATH = r"C:\Users\Sree\OneDrive\Desktop\faang\standard_scaler.pkl"
ENCODER_PATH = r"C:\Users\Sree\OneDrive\Desktop\faang\company_encoder.pkl"


# --------------------------
# Helper: safe load
# --------------------------
def safe_load(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        st.warning(f"Failed to load {p.name}: {e}")
        return None


model = safe_load(MODEL_PATH)
feature_columns = safe_load(FEATURE_COLS_PATH)
scaler = safe_load(SCALER_PATH)
company_encoder = safe_load(ENCODER_PATH)


# --------------------------
# Company mappings + logo URLs
# --------------------------
COMPANY_CHOICES = {
    "Apple (AAPL)": "AAPL",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "Google (GOOGL)": "GOOGL",
    "Netflix (NFLX)": "NFLX",
}
TICKER_TO_NAME = {"AAPL": "Apple", "AMZN": "Amazon", "META": "Meta",
                  "GOOGL": "Google", "NFLX": "Netflix"}

LOGO_URLS = {
    "AAPL": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
    "AMZN": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
    "META": r"C:\Users\Sree\OneDrive\Desktop\faang\facebook-logo.png",
    "GOOGL": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
    "NFLX": "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
}


# --------------------------
# Sidebar Inputs
# --------------------------
with st.sidebar.expander("📊 Input Parameters", expanded=True):

    company_display = st.selectbox("Select Company", list(COMPANY_CHOICES.keys()))
    ticker = COMPANY_CHOICES[company_display]

    try:
        st.image(LOGO_URLS.get(ticker), width=64)
    except:
        st.write(TICKER_TO_NAME.get(ticker))

    st.markdown("---")

    open_price = st.number_input("Open Price", min_value=0.0, value=150.0, step=0.5)
    high_price = st.number_input("High Price", min_value=0.0, value=155.0, step=0.5)
    low_price = st.number_input("Low Price", min_value=0.0, value=145.0, step=0.5)
    volume = st.number_input("Volume", min_value=0, value=1_000_000, step=1000)

    selected_date = st.date_input("Date", value=date.today())
    year, month, day = selected_date.year, selected_date.month, selected_date.day

    optional_fields = ["Market Cap", "PE Ratio", "EPS", "Profit Margin",
                       "Return on Equity (ROE)", "Debt to Equity"]
    optional_values = {}

    if feature_columns:
        for f in optional_fields:
            if f in feature_columns:
                optional_values[f] = st.number_input(f, value=0.0, step=0.1)


# --------------------------
# MLflow Sidebar
# --------------------------
with st.sidebar.expander("📌 MLflow Tracking Information", expanded=False):
    st.markdown("*Best Final Model*")
    st.markdown("[Run (best_final_model)](https://dagshub.com/tstr12cg429/my-first-repo.mlflow/#/experiments/0/runs/17ee9098dda8431797ae99356e0c09bf)")
    st.markdown("---")
    st.markdown("*Experiment Runs*")

    runs = [
        ("Linear Regression", "1ce18e6b6ca24057b7adf4755ddceb41", 1.0000),
        ("Decision Tree", "ada275a5e4404809a35fc3f74d11a481", 0.9761),
        ("Random Forest", "8fb1cb08076541599f60bf14939257af", 0.9754),
        ("Gradient Boosting", "df9dce98256f4bbeb00265ea49e16a2f", 0.9888),
        ("SVR", "17263dfb22b2467499314db36ada8c64", 0.5453),
        ("XGBoost", "1184c16cad3a48a58c0333991ade5e54", 0.9725),
    ]

    for name, rid, r2 in runs:
        st.markdown(f"- *{name}* — R² = {r2:.4f} — "
                    f"[Run link](https://dagshub.com/tstr12cg429/my-first-repo.mlflow/#/experiments/0/runs/{rid})")

    st.markdown("---")
    st.markdown("[Open MLflow Dashboard](https://dagshub.com/tstr12cg429/my-first-repo.mlflow/#/experiments/0)")


# --------------------------
# Artifact Status
# --------------------------
with st.sidebar.expander("🗂 Artifacts & Tracking", expanded=False):
    st.write("Model:", "✅ Loaded" if model is not None else "❌ Missing")
    st.write("Feature columns:", "✅ Loaded" if feature_columns is not None else "❌ Missing")
    st.write("Scaler:", "✅ Loaded" if scaler is not None else "❌ Missing")
    st.write("Company encoder:", "✅ Loaded" if company_encoder is not None else "❌ Missing")


st.markdown("<hr/>", unsafe_allow_html=True)


# --------------------------
# Build Input Vector
# --------------------------
def build_input_vector(inputs: dict):
    if feature_columns is None:
        st.error("feature_columns.pkl not loaded.")
        return None

    X = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    for k, v in inputs.items():
        if k in X.columns:
            X.at[0, k] = v

    if "Company_encoded" in X.columns and company_encoder:
        try:
            X.at[0, "Company_encoded"] = int(
                company_encoder.transform([inputs["Company"]])[0])
        except:
            try:
                X.at[0, "Company_encoded"] = int(
                    company_encoder.transform([TICKER_TO_NAME.get(inputs["Ticker"])]))[0]
            except:
                pass

    for col in [c for c in feature_columns if c.startswith("Ticker_")]:
        if col.endswith(inputs["Ticker"]):
            X.at[0, col] = 1

    if scaler:
        try:
            X = pd.DataFrame(scaler.transform(X), columns=feature_columns)
        except Exception as e:
            st.warning(f"Scaler.transform failed: {e}")

    return X


# --------------------------
# App Tabs
# --------------------------
tabs = st.tabs(["Prediction", "Historical Data",
                "Model Comparison", "Explainability", "Downloads"])


# --------------------------
# TAB 1 — Prediction
# --------------------------
with tabs[0]:
    st.header("🚀 Predict Closing Price")

    st.write("""
Adjust the stock parameters from the sidebar to generate a prediction.
Your inputs are automatically processed and scaled using the same pipeline used during training.
Click **Predict Now** to compute the expected closing price.
""")

    if st.button("Predict Now"):
        if model is None or feature_columns is None:
            st.error("Model or feature_columns missing.")
        else:
            inputs = {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Volume": volume,
                "Year": year,
                "Month": month,
                "Day": day,
                "Company": TICKER_TO_NAME[ticker],
                "Ticker": ticker,
            }

            inputs.update(optional_values)

            X_in = build_input_vector(inputs)

            try:
                pred = model.predict(X_in)[0]
                st.success(f"📌 Predicted Close Price: **${pred:,.2f}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# --------------------------
# TAB 2 — Historical Data
# --------------------------
with tabs[1]:
    st.header("📈 Historical Data")
    if st.button("Load historical data"):
        hist = yf.download(ticker, period="3y", progress=False)
        if hist.empty:
            st.warning("No data returned.")
        else:
            hist = hist.reset_index()
            st.dataframe(hist.head())
            st.line_chart(hist.set_index("Date")["Close"])


# --------------------------
# TAB 3 — Model Comparison
# --------------------------
with tabs[2]:
    st.header("📊 Model Comparison")

    comp_df = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest",
                  "Gradient Boosting", "SVR", "XGBoost"],
        "MAE": [0.5, 2.5, 3.0, 1.2, 54.488, 10.0],
        "R2": [1.0000, 0.9761, 0.9754, 0.9888, 0.5453, 0.9725],
        "RMSE": [0.3, 12.0, 18.0, 8.0, 106.1, 20.0]
    })

    st.dataframe(comp_df)


# --------------------------
# TAB 4 — SHAP
# --------------------------
with tabs[3]:
    st.header("🔬 Explainability (SHAP)")

    try:
        import shap
        shap.initjs()
    except:
        st.warning("SHAP is not installed. Run: pip install shap")
        st.stop()

    uploaded = st.file_uploader("Upload CSV for SHAP", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded).reindex(columns=feature_columns, fill_value=0)
            df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_columns)

            explainer = shap.Explainer(model)
            shap_values = explainer(df_scaled)

            st.subheader("Summary Plot")
            st.pyplot(shap.plots.beeswarm(shap_values, show=False))

        except Exception as e:
            st.error(f"SHAP failed: {e}")


# --------------------------
# TAB 5 — Downloads
# --------------------------
with tabs[4]:
    st.header("⬇ Downloads")

    if Path(MODEL_PATH).exists():
        st.download_button("Download Model", open(MODEL_PATH, "rb").read(),
                           file_name="best_model.pkl")

    if Path(FEATURE_COLS_PATH).exists():
        st.download_button("Download feature_columns.pkl",
                           open(FEATURE_COLS_PATH, "rb").read(),
                           file_name="feature_columns.pkl")

    if feature_columns:
        sample_template = pd.DataFrame(columns=feature_columns)
        st.download_button("Download Input Template",
                           sample_template.to_csv(index=False).encode(),
                           file_name="input_template.csv")

st.markdown("---")
st.caption("✨ Built by *Sree V G* — FAANG ML Model • Streamlit • MLflow")
