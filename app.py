import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.model import load_data, train_model, predict_risk

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Healthcare Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;"> Healthcare Risk Score Prediction</h1>
    <p style="text-align:center; font-size:18px;">
    Explainable AI model for healthcare risk assessment
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Load Data & Train Model (Cached)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_train():
    df = load_data()
    model = train_model(df)
    return df, model

df, model = load_and_train()

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("🧑‍⚕️ Patient Information")

age = st.sidebar.slider("Age (years)", 18, 80, 40)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 90, 180, 120)
chol = st.sidebar.slider("Cholesterol (mg/dL)", 150, 300, 200)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 70, 200, 100)
smoking = st.sidebar.radio("Smoking Habit", ["No", "Yes"])

smoking_val = 1 if smoking == "Yes" else 0

if bmi < 10 or bmi > 60:
    st.sidebar.warning("BMI value looks unusual. Please verify.")

# --------------------------------------------------
# Layout
# --------------------------------------------------
col1, col2 = st.columns(2)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.sidebar.button(" Predict Risk"):
    patient = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "BloodPressure": [bp],
        "Cholesterol": [chol],
        "Glucose": [glucose],
        "Smoking": [smoking_val]
    })

    risk = predict_risk(model, patient)[0]

    # -------- Left Column: Result --------
    with col1:
        st.subheader(" Prediction Result")
        st.metric("Predicted Risk Score", f"{risk:.2f}")

        if risk < 50:
            st.success("🟢 Low Health Risk")
        elif risk < 70:
            st.warning("🟡 Moderate Health Risk")
        else:
            st.error("🔴 High Health Risk")

        st.markdown("### Risk Level Indicator")
        st.progress(min(int(risk), 100))

        st.info(
            "This prediction is based on historical data patterns. "
            "Clinical decisions should involve medical professionals."
        )

    # -------- Right Column: Feature Importance --------
    with col2:
        st.subheader("📈 Feature Importance")

        importance = pd.DataFrame({
            "Feature": df.drop("RiskScore", axis=1).columns,
            "Impact": model.coef_
        }).sort_values(by="Impact")

        fig, ax = plt.subplots()
        ax.barh(importance["Feature"], importance["Impact"])
        ax.set_xlabel("Impact on Risk Score")
        ax.set_title("Linear Regression Coefficients")

        st.pyplot(fig)

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
with st.expander("📂 View Sample Dataset"):
    st.dataframe(df.head(10))

# --------------------------------------------------
# About Section
# --------------------------------------------------
with st.expander(" About This Application"):
    st.write("""
    - **Model:** Linear Regression  
    - **Purpose:** Healthcare risk modeling  
    - **Inputs:** Patient vitals and lifestyle factors  
    - **Output:** Continuous risk score  

     Educational use only. Not for medical diagnosis.
    """)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Built with  using Streamlit & Linear Regression</div>",
    unsafe_allow_html=True
)
