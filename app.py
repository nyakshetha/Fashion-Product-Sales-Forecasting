"""
app.py
------
Streamlit Web Application for Fitting Room Garment Purchase Prediction.
Provides an interactive UI to input product attributes and fitting room trial counts,
predicting whether a garment will be purchased or not using the trained XGBoost model.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Fitting Room Purchase Predictor",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS for Premium Styling
# ============================================================
st.markdown("""
<style>
    /* ---------- Google Font Import ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Main Container ---------- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ---------- Header ---------- */
    .main-header {
        text-align: center;
        padding: 2.5rem 1.5rem 1.5rem 1.5rem;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(48, 43, 99, 0.35);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #c4b5fd;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }
    .main-header .emoji-icon {
        font-size: 2.8rem;
        margin-bottom: 0.4rem;
        display: block;
    }

    /* ---------- Section Cards ---------- */
    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: box-shadow 0.3s ease;
    }
    .section-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .section-card h3 {
        color: #1e1b4b;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #e0e7ff;
    }

    /* ---------- Result Boxes ---------- */
    .result-box {
        text-align: center;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        animation: fadeInUp 0.6s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-purchased {
        background: linear-gradient(135deg, #065f46, #059669);
        border: 2px solid #34d399;
        box-shadow: 0 8px 30px rgba(5, 150, 105, 0.3);
    }
    .result-not-purchased {
        background: linear-gradient(135deg, #991b1b, #dc2626);
        border: 2px solid #f87171;
        box-shadow: 0 8px 30px rgba(220, 38, 38, 0.3);
    }
    .result-box h2 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .result-box p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0;
    }

    /* ---------- Probability Badge ---------- */
    .prob-badge {
        display: inline-block;
        padding: 0.6rem 1.8rem;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.8rem;
    }
    .prob-high   { background: linear-gradient(90deg, #059669, #10b981); }
    .prob-medium { background: linear-gradient(90deg, #d97706, #f59e0b); }
    .prob-low    { background: linear-gradient(90deg, #dc2626, #ef4444); }

    /* ---------- Metric Cards ---------- */
    .metric-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 160px;
        flex: 1;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 0.2rem;
    }


    /* ---------- Button ---------- */
    .stButton > button {
        width: 100%;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #ffffff;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.35);
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #6d28d9);
        box-shadow: 0 6px 25px rgba(79, 70, 229, 0.5);
        transform: translateY(-1px);
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        color: #94a3b8;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
    }
    .footer strong { color: #64748b; }

    /* ---------- Input Summary Table ---------- */
    .input-summary {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
    }
    .input-summary table {
        width: 100%;
        border-collapse: collapse;
    }
    .input-summary td {
        padding: 0.35rem 0.6rem;
        font-size: 0.9rem;
    }
    .input-summary td:first-child {
        color: #64748b;
        font-weight: 600;
        width: 45%;
    }
    .input-summary td:last-child {
        color: #1e293b;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Constants — Training Feature Columns (must match model)
# ============================================================
TRAINING_FEATURES = [
    "Price", "Trial_Count",
    "Size_l", "Size_m", "Size_s", "Size_xl",
    "Sleeve_Type_full", "Sleeve_Type_half", "Sleeve_Type_sleeveless",
    "Color_black", "Color_blue", "Color_green",
    "Color_red", "Color_white", "Color_yellow",
    "Collar_Type_mandarin", "Collar_Type_polo",
    "Collar_Type_round", "Collar_Type_v-neck",
    "Fit_loose", "Fit_regular", "Fit_slim",
    "Pattern_checked", "Pattern_printed",
    "Pattern_solid", "Pattern_striped",
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "purchase_model.joblib")
FEATURE_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "feature_importance.png")


# ============================================================
# Helper Functions
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained XGBoost model from disk (cached)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def prepare_input(size, sleeve, color, collar, fit, pattern, price, trial_count):
    """
    Convert user input into a DataFrame aligned with training features.

    The approach replicates the preprocessing pipeline:
      1. Build a single-row DataFrame with raw values (lowercased).
      2. Apply pd.get_dummies to one-hot encode categorical columns.
      3. Reindex to match training feature columns, filling missing
         columns with 0.
    """
    # --- Build raw input DataFrame ---
    raw = pd.DataFrame([{
        "Size": size.lower().strip(),
        "Sleeve_Type": sleeve.lower().strip(),
        "Color": color.lower().strip(),
        "Collar_Type": collar.lower().strip(),
        "Fit": fit.lower().strip(),
        "Pattern": pattern.lower().strip(),
        "Price": price,
        "Trial_Count": trial_count,
    }])

    # --- One-hot encode categorical columns ---
    categorical_cols = ["Size", "Sleeve_Type", "Color", "Collar_Type", "Fit", "Pattern"]
    encoded = pd.get_dummies(raw, columns=categorical_cols, drop_first=False)

    # --- Convert boolean columns to int ---
    bool_cols = encoded.select_dtypes(include=["bool"]).columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)

    # --- Align with training features (add missing cols as 0) ---
    aligned = encoded.reindex(columns=TRAINING_FEATURES, fill_value=0)

    return aligned




# ============================================================
# Main Header
# ============================================================
st.markdown("""
<div class="main-header">
    <span class="emoji-icon">👗</span>
    <h1>Fitting Room Garment Purchase Prediction</h1>
    <p>Predict customer purchase intent based on fitting room trial dynamics and garment physical attributes</p>
</div>
""", unsafe_allow_html=True)

st.info("🏬 **Fitting Room Analytics Insight**: This model is trained on in-store **Fitting Room Interaction Data**. Fitting room trial frequency (`Trial_Count`) combines with physical attributes (Size, Fit, Sleeve, Pattern) and pricing to forecast purchase conversion probability.")


# ============================================================
# Load Model
# ============================================================
model = load_model()

if model is None:
    st.error(
        "⚠️ **Model file not found!**  \n"
        f"Expected path: `{MODEL_PATH}`  \n"
        "Please run `python main.py` first to train and save the model."
    )
    st.stop()


# ============================================================
# Input Form
# ============================================================
st.markdown('<div class="section-card"><h3>📝 Enter Garment & Fitting Room Details</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    size = st.selectbox("👕 Size", options=["S", "M", "L", "XL"], index=1)

with col2:
    sleeve = st.selectbox("🧥 Sleeve Type", options=["Full", "Half", "Sleeveless"], index=0)

with col3:
    color = st.selectbox("🎨 Color", options=["Black", "Blue", "White", "Red", "Green", "Yellow"], index=0)

with col4:
    collar = st.selectbox("👔 Collar Type", options=["Round", "Polo", "V-Neck", "Mandarin"], index=0)

col5, col6, col7, col8 = st.columns(4)

with col5:
    fit = st.selectbox("📐 Fit", options=["Slim", "Regular", "Loose"], index=1)

with col6:
    pattern = st.selectbox("🔲 Pattern", options=["Solid", "Printed", "Striped", "Checked"], index=0)

with col7:
    price = st.slider("💰 Price (₹)", min_value=200, max_value=3000, value=800, step=50)

with col8:
    trial_count = st.slider("🔄 Fitting Room Trial Count", min_value=0, max_value=10, value=2, step=1, help="Number of times customer tried on the garment in the fitting room")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Predict Button
# ============================================================
predict_clicked = st.button("🔮  Predict Purchase Likelihood", use_container_width=True)


# ============================================================
# Prediction Logic & Output
# ============================================================
CUSTOM_THRESHOLD = 0.35  # Tuned threshold (default 0.5 is too strict)

if predict_clicked:
    try:
        # --- Prepare input ---
        input_df = prepare_input(size, sleeve, color, collar, fit, pattern, price, trial_count)

        # --- Get probability scores ---
        probability = model.predict_proba(input_df)[0]
        proba = probability[1]  # Purchase probability

        # --- Debug print to console ---
        print(f"[DEBUG] Purchase Probability: {proba:.4f} | Threshold: {CUSTOM_THRESHOLD}")

        # --- Apply custom threshold ---
        prediction = 1 if proba >= CUSTOM_THRESHOLD else 0

        prob_purchased = proba * 100
        prob_not_purchased = probability[0] * 100

        # --- Determine interpretation level ---
        if proba < 0.3:
            interpretation = "🔴 Very Low Purchase Probability"
            interp_color = "#dc2626"
        elif proba < 0.5:
            interpretation = "🟠 Low Purchase Probability"
            interp_color = "#f59e0b"
        elif proba < 0.7:
            interpretation = "🟡 Moderate Purchase Probability"
            interp_color = "#eab308"
        else:
            interpretation = "🟢 High Purchase Probability"
            interp_color = "#059669"

        # --- Display Results ---
        st.markdown("---")

        if prediction == 1:
            st.markdown(f"""
            <div class="result-box result-purchased">
                <h2>✅ Likely to be Purchased</h2>
                <p>The model predicts this product <strong>will be purchased</strong></p>
                <div class="prob-badge prob-high">{prob_purchased:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-not-purchased">
                <h2>❌ Not Likely to be Purchased</h2>
                <p>The model predicts this product <strong>will not be purchased</strong></p>
                <div class="prob-badge prob-low">{prob_not_purchased:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        # --- Interpretation Badge ---
        st.markdown(f"""
        <div style="text-align:center; margin: 0.8rem 0;">
            <span style="background:{interp_color}22; color:{interp_color}; border:1px solid {interp_color}44;
                         padding:0.5rem 1.2rem; border-radius:50px; font-weight:700; font-size:0.95rem;">
                {interpretation}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # --- Detailed Metrics ---
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Purchase Probability</div>
                <div class="metric-value" style="color: #059669;">{prob_purchased:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Non-Purchase Probability</div>
                <div class="metric-value" style="color: #dc2626;">{prob_not_purchased:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value">{"Purchased" if prediction == 1 else "Not Purchased"}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Threshold</div>
                <div class="metric-value" style="font-size: 1.1rem;">{CUSTOM_THRESHOLD}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Input Summary ---
        st.markdown("")
        with st.expander("📋 View Input Summary", expanded=False):
            st.markdown(f"""
            <div class="input-summary">
            <table>
                <tr><td>Size</td><td>{size}</td></tr>
                <tr><td>Sleeve Type</td><td>{sleeve}</td></tr>
                <tr><td>Color</td><td>{color}</td></tr>
                <tr><td>Collar Type</td><td>{collar}</td></tr>
                <tr><td>Fit</td><td>{fit}</td></tr>
                <tr><td>Pattern</td><td>{pattern}</td></tr>
                <tr><td>Price</td><td>₹{price}</td></tr>
                <tr><td>Fitting Room Trial Count</td><td>{trial_count}</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ **Prediction Error:** {str(e)}")


# ============================================================
# Feature Importance Section
# ============================================================
st.markdown("---")
st.markdown('<div class="section-card"><h3>📊 Feature Importance Analysis</h3>', unsafe_allow_html=True)

if os.path.exists(FEATURE_IMG_PATH):
    st.image(FEATURE_IMG_PATH, caption="Top garment physical attributes and fitting room dynamics influencing purchase prediction")
else:
    st.info("Feature importance plot not available. Run `python main.py` to generate it.")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="footer">
    <strong>Fitting Room Garment Purchase Prediction System</strong><br>
    In-Store Retail Analytics &nbsp;|&nbsp; Powered by XGBoost & Streamlit
</div>
""", unsafe_allow_html=True)
