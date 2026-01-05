import pickle
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st


# =============================
# Page configuration
# =============================
st.set_page_config(page_title="Trymore's Churn Prediction System", layout="wide")
st.title("📉 Customer Churn Prediction System")
st.caption("Logistic Regression vs Random Forest")


# =============================
# Artifact loading
# =============================
MODELS_DIR = Path("models")


def _safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_pipelines() -> Tuple[object, object]:
    logistic_pipeline = _safe_load(MODELS_DIR / "logistic_pipeline.pkl")
    rf_pipeline = _safe_load(MODELS_DIR / "rf_pipeline.pkl")
    return logistic_pipeline, rf_pipeline


# =============================
# Sidebar
# =============================
def sidebar_controls() -> Tuple[str, float]:
    st.sidebar.header("⚙️ Configuration")

    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest"]
    )

    threshold = st.sidebar.slider(
        "Churn Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    return model_choice, threshold


# =============================
# Input Form
# =============================
def build_input_form() -> Optional[pd.DataFrame]:
    st.subheader("🧾 Customer Information")

    with st.form("customer_form"):
        # Demographics
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with d2:
            senior = st.selectbox("Senior Citizen", [0, 1])
        with d3:
            partner = st.selectbox("Partner", ["Yes", "No"])
        with d4:
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        # Billing
        st.subheader("💳 Billing")
        b1, b2, b3 = st.columns(3)
        with b1:
            tenure = st.number_input("Tenure (months)", min_value=0, value=1)
        with b2:
            monthly = st.number_input(
                "Monthly Charges", min_value=0.0, value=50.0
            )
        with b3:
            auto_total = st.checkbox("Auto-calc Total Charges", value=True)
            if auto_total:
                total = round(tenure * monthly, 2)
                st.write(f"Total Charges: {total}")
            else:
                total = st.number_input(
                    "Total Charges", min_value=0.0, value=0.0
                )

        # Services
        with st.expander("📡 Services"):
            s1, s2, s3 = st.columns(3)
            with s1:
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiple = st.selectbox(
                    "Multiple Lines", ["Yes", "No", "No phone service"]
                )
                internet = st.selectbox(
                    "Internet Service", ["DSL", "Fiber optic", "No"]
                )
            with s2:
                online_sec = st.selectbox(
                    "Online Security", ["Yes", "No", "No internet service"]
                )
                online_backup = st.selectbox(
                    "Online Backup", ["Yes", "No", "No internet service"]
                )
                device_protect = st.selectbox(
                    "Device Protection", ["Yes", "No", "No internet service"]
                )
            with s3:
                tech_support = st.selectbox(
                    "Tech Support", ["Yes", "No", "No internet service"]
                )
                streaming_tv = st.selectbox(
                    "Streaming TV", ["Yes", "No", "No internet service"]
                )
                streaming_movies = st.selectbox(
                    "Streaming Movies", ["Yes", "No", "No internet service"]
                )

        # Contract
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
        with c2:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with c3:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        submit = st.form_submit_button("🔮 Predict Churn")

    if not submit:
        return None

    return pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])



    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    # Logistic Regression → coefficients
    if hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.coef_[0]
        })
        coef_df["Abs"] = coef_df["Coefficient"].abs()
        return (
            coef_df.sort_values("Abs", ascending=False)
                   .drop(columns="Abs")
                   .head(top_n)
                   .set_index("Feature")
        )

    # Random Forest → feature importance
    if hasattr(model, "feature_importances_"):
        return (
            pd.Series(
                model.feature_importances_,
                index=feature_names
            )
            .sort_values(ascending=False)
            .head(top_n)
            .to_frame("Importance")
        )

    return None


# =============================
# Main
# =============================
def main():
    logistic_pipeline, rf_pipeline = load_pipelines()
    model_choice, threshold = sidebar_controls()

    pipeline = (
        logistic_pipeline
        if model_choice == "Logistic Regression"
        else rf_pipeline
    )

    input_df = build_input_form()
    if input_df is None:
        return

    churn_prob = pipeline.predict_proba(input_df)[0][1]
    churn_pred = int(churn_prob >= threshold)

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{churn_prob:.2%}")
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Threshold:** {threshold}")

    if churn_pred:
        st.error("⚠️ High Risk: Customer Likely to Churn")
    else:
        st.success("✅ Low Risk: Customer Likely to Stay")

    with st.expander("📈 Model Explanation"):
        st.write(
            "Logistic Regression: feature coefficients (directional impact).  \n"
            "Random Forest: global feature importance."
        )

       

if __name__ == "__main__":
    main()
