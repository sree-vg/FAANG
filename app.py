# app.py — Final Streamlit FAANG Dashboard (global banner + sidebar ordering)
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
SCALER_PATH = r"C:\Users\Sree\OneDrive\Desktop\faang\minmax_scaler.pkl"
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
        # don't crash the app if pickle fails
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
TICKER_TO_NAME = {"AAPL":"Apple","AMZN":"Amazon","META":"Meta","GOOGL":"Google","NFLX":"Netflix"}

# Some common logo URLs (online). If blocked, fallback to text.
LOGO_URLS = {
    "AAPL": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
    "AMZN": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",

    # META → Try URL first, fallback to local static file
    "META": r"C:\Users\Sree\OneDrive\Desktop\faang\facebook-logo.png",

    "GOOGL": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
    "NFLX": "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
}

# FAANG banner (horizontal) — if these remote URLs are blocked, banner falls back to text.
BANNER_URLS = [
    LOGO_URLS["AAPL"],
    LOGO_URLS["AMZN"],
    LOGO_URLS["META"],
    LOGO_URLS["GOOGL"],
    LOGO_URLS["NFLX"],
]

# Sidebar layout: we will put Input Parameters first,
# then MLflow Tracking Information, then Artifact status — as requested
# --------------------------

# 1) INPUT PARAMETERS (expanded)
with st.sidebar.expander("📊 Input Parameters", expanded=True):
    company_display = st.selectbox("Select Company", list(COMPANY_CHOICES.keys()))
    ticker = COMPANY_CHOICES[company_display]

    # show dynamic logo next to selectbox (small)
    try:
        st.image(LOGO_URLS.get(ticker), width=64)
    except Exception:
        st.write(f"{TICKER_TO_NAME.get(ticker, ticker)}")

    st.markdown("---")
    open_price = st.number_input("Open Price", min_value=0.0, value=150.0, step=0.5)
    high_price = st.number_input("High Price", min_value=0.0, value=155.0, step=0.5)
    low_price = st.number_input("Low Price", min_value=0.0, value=145.0, step=0.5)
    volume = st.number_input("Volume", min_value=0, value=1_000_000, step=1000)
    selected_date = st.date_input("Date", value=date.today())
    year, month, day = selected_date.year, selected_date.month, selected_date.day

    # optional extra fields (only displayed if present in feature_columns)
    optional_fields = ["Market Cap", "PE Ratio", "EPS", "Profit Margin", "Return on Equity (ROE)", "Debt to Equity"]
    optional_values = {}
    if feature_columns:
        for f in optional_fields:
            if f in feature_columns:
                optional_values[f] = st.number_input(f, value=0.0, step=0.1)

# 2) MLflow Tracking Information (collapsed by default)
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
        st.markdown(f"- *{name}* — R² = {r2:.4f}  —  [Run link](https://dagshub.com/tstr12cg429/my-first-repo.mlflow/#/experiments/0/runs/{rid})")
    st.markdown("---")
    st.markdown("[Open MLflow Experiments Dashboard](https://dagshub.com/tstr12cg429/my-first-repo.mlflow/#/experiments/0)")

# 3) Artifacts & Tracking (collapsed by default)
with st.sidebar.expander("🗂 Artifacts & Tracking", expanded=False):
    st.write("Model:", "✅ Loaded" if model is not None else "❌ Missing")
    st.write("Feature columns:", "✅ Loaded" if feature_columns is not None else "❌ Missing")
    st.write("Scaler:", "✅ Loaded" if scaler is not None else "❌ Missing")
    st.write("Company encoder:", "✅ Loaded" if company_encoder is not None else "❌ Missing")

st.markdown("<hr/>", unsafe_allow_html=True)

# --------------------------
# Utility: build input vector aligned to feature_columns
# --------------------------
def build_input_vector(inputs: dict):
    if feature_columns is None:
        st.error("feature_columns.pkl not loaded — cannot build input vector.")
        return None
    X = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    # assign numeric inputs
    for k, v in inputs.items():
        if k in X.columns:
            X.at[0, k] = v
    # company encoder
    if "Company_encoded" in X.columns and company_encoder:
        try:
            # encoder may expect full company name
            X.at[0, "Company_encoded"] = int(company_encoder.transform([inputs["Company"]])[0])
        except Exception:
            # try ticker name fallback
            try:
                X.at[0, "Company_encoded"] = int(company_encoder.transform([TICKER_TO_NAME.get(inputs["Ticker"], inputs["Company"])])[0])
            except Exception:
                pass
    # one-hot ticker
    ticker_cols = [c for c in feature_columns if c.startswith("Ticker_")]
    if "Ticker" in inputs:
        for c in ticker_cols:
            if c.endswith(inputs["Ticker"]):
                X.at[0, c] = 1
    # optional extras already set above
    # scaling
    if scaler is not None:
        try:
            scaled = scaler.transform(X)
            X = pd.DataFrame(scaled, columns=feature_columns)
        except Exception as e:
            st.warning(f"Scaler.transform failed: {e}. Using unscaled features.")
    return X

# --------------------------
# Layout: Tabs
# --------------------------
tabs = st.tabs(["Prediction", "Historical Data", "Model Comparison", "Explainability", "Downloads"])

# ----- TAB 1: Prediction -----
with tabs[0]:
    st.header("🚀 Predict Closing Price")
    st.write("Select your stock parameters from the sidebar. Once everything is set, click Predict Now to generate the estimated closing price using the trained ML model.")
    # single button here only; inputs are in sidebar as requested
    if st.button("Predict Now"):
        if model is None or feature_columns is None:
            st.error("Missing model or feature_columns.pkl — cannot predict.")
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
            # include any optional values (if present)
            inputs.update(optional_values)
            X_in = build_input_vector(inputs)
            if X_in is None:
                st.error("Failed to build input vector.")
            else:
                try:
                    pred = model.predict(X_in)[0]
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    pred = None
                if pred is not None:
                    st.success(f"📌 Predicted Close Price: *${float(pred):,.2f}*")
                    st.subheader("Feature Vector used (post-scaling if scaler loaded)")
                    st.dataframe(X_in.T)
                    # download button
                    outdf = X_in.copy()
                    outdf["Predicted_Close"] = pred
                    st.download_button("⬇ Download prediction (CSV)", data=outdf.to_csv(index=False).encode("utf-8"), file_name="prediction_vector.csv", mime="text/csv")

# ----- TAB 2: Historical Data -----
with tabs[1]:
    st.header("📈 Historical Data (from Yahoo Finance)")
    st.write("Click *Load historical data* to fetch 3 years of data for the selected ticker (yfinance).")
    if st.button("Load historical data"):
        try:
            hist = yf.download(ticker, period="3y", progress=False)
            if hist is None or hist.empty:
                st.warning("No data returned from yfinance.")
            else:
                hist = hist.reset_index()
                st.success(f"Loaded {hist.shape[0]} rows")
                st.dataframe(hist.head())
                # show close chart
                st.subheader("Close Price Over Time")
                st.line_chart(hist.set_index("Date")["Close"])
                # show volumes as well
                if "Volume" in hist.columns:
                    st.subheader("Volume Over Time")
                    st.area_chart(hist.set_index("Date")["Volume"])
        except Exception as e:
            st.error(f"Failed to fetch historical data: {e}")

# ----- TAB 3: Model Comparison -----
with tabs[2]:
    st.header("📊 Model Comparison")
    st.write("Below is the model comparison visualization from your local file.")

    # Path to your local model comparison image
    MODEL_COMP_PATH = r"C:\Users\Sree\OneDrive\Desktop\faang\Model_comparison.png"

    # Display the table
    comp_df = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR", "XGBoost"],
        "MAE": [0.5, 2.5, 3.0, 1.2, 54.488, 10.0],
        "R2": [1.0000, 0.9761, 0.9754, 0.9888, 0.5453, 0.9725],
        "RMSE": [0.3, 12.0, 18.0, 8.0, 106.1, 20.0]
    })
    st.dataframe(comp_df.style.format({"MAE": "{:.3f}", "R2": "{:.4f}", "RMSE": "{:.3f}"}))

    st.markdown("### 📌 Model Comparison Chart")
    try:
        st.image(MODEL_COMP_PATH, use_container_width=True)
    except:
        st.error("Could not load model comparison image. Check the file path.")


# ----- TAB 4: Explainability (SHAP) -----
# ----- TAB 4: Explainability (SHAP) -----
with tabs[3]:
    st.header("🔬 Explainability (SHAP)")

    # Check if SHAP is installed
    try:
        import shap
        shap.initjs()
        shap_ok = True
    except:
        shap_ok = False

    if not shap_ok:
        st.warning("⚠️ SHAP not installed. Run:  pip install shap")
        st.stop()

    st.write("Upload a CSV containing the SAME feature columns as your model uses.")

    uploaded = st.file_uploader("Upload CSV for SHAP Analysis", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)

            # Ensure correct column order
            if feature_columns is None:
                st.error("feature_columns.pkl missing — cannot compute SHAP.")
                st.stop()

            df = df.reindex(columns=feature_columns, fill_value=0)

            # Apply scaling if available
            if scaler is not None:
                df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_columns)
            else:
                df_scaled = df

            st.success("File loaded successfully!")

            # ---------------------------
            # ⭐ SHAP OPTION A (Local Explainability)
            # ---------------------------
            st.subheader("📌 SHAP Summary Plot")

            explainer = shap.Explainer(model)
            shap_values = explainer(df_scaled)

            fig1 = shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig1)

            # Bar plot
            st.subheader("📊 Feature Importance (SHAP Bar Plot)")
            fig2 = shap.plots.bar(shap_values, show=False)
            st.pyplot(fig2)

            # Waterfall for 1st row
            st.subheader("🧩 Waterfall Plot (Row 1 Explanation)")
            fig3 = shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"Failed to compute SHAP values: {e}")


# ----- TAB 5: Downloads -----
with tabs[4]:
    st.header("⬇ Downloads")
    if Path(MODEL_PATH).exists():
        st.download_button("Download model (best_model.pkl)", data=open(MODEL_PATH,"rb").read(), file_name="best_model.pkl")
    else:
        st.info("best_model.pkl not found at configured path.")
    if Path(FEATURE_COLS_PATH).exists():
        st.download_button("Download feature_columns.pkl", data=open(FEATURE_COLS_PATH,"rb").read(), file_name="feature_columns.pkl")
    else:
        st.info("feature_columns.pkl not found at configured path.")

    if feature_columns is not None:
        sample_template = pd.DataFrame(columns=feature_columns)
        st.download_button("Download sample input template (CSV)", data=sample_template.to_csv(index=False).encode("utf-8"), file_name="sample_input_template.csv", mime="text/csv")

st.markdown("---")
st.caption("✨ Built by *Sree V G* — FAANG ML Model • Streamlit • MLflow")
