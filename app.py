import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-top: 5px;
        margin-bottom: 30px;
    }

    .info-card {
        padding: 18px;
        border-radius: 14px;
        background-color: rgba(128,128,128,0.08);
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD ARTIFACTS (cached — loads once per session, not per rerun)
# ============================================================

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        BASE_DIR / "model.h5",
        compile=False,
    )

    with open(BASE_DIR / "one_hot_encoder_geography.pkl", "rb") as file:
        geography_encoder = pickle.load(file)

    with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as file:
        gender_encoder = pickle.load(file)

    with open(BASE_DIR / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, geography_encoder, gender_encoder, scaler


try:
    model, geography_encoder, gender_encoder, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not load one or more model artifacts "
        f"(model.h5 / *.pkl). Missing file: {e.filename}"
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Churn Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Predict customer churn risk using an Artificial Neural Network
    and identify customers who may need proactive retention attention.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** Artificial Neural Network")
    st.write("**Task:** Binary Classification")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Architecture:** 12 → 64 → 32 → 1")
    st.write("**Output:** Churn Probability")
    st.write("**Decision Threshold:** 0.50")

    st.divider()

    st.caption(
        "This application is a decision-support tool. "
        "Predictions should be interpreted alongside business context, "
        "not treated as a guaranteed outcome."
    )


# ============================================================
# INPUT FORM
# ============================================================

with st.form("customer_prediction_form"):
    profile_col, financial_col = st.columns(2)

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------
    with profile_col:
        st.subheader("Customer Profile")

        geography = st.selectbox(
            "Geography",
            geography_encoder.categories_[0],
        )

        gender = st.selectbox(
            "Gender",
            gender_encoder.classes_,
        )

        age = st.slider("Age", min_value=18, max_value=92, value=35)

        tenure = st.slider("Tenure (years)", min_value=0,
                           max_value=10, value=5)

        num_of_products = st.slider(
            "Number of Products", min_value=1, max_value=4, value=2
        )

    # --------------------------------------------------------
    # FINANCIAL / ACCOUNT PROFILE
    # --------------------------------------------------------
    with financial_col:
        st.subheader("Financial & Account Details")

        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=650, step=1
        )

        balance = st.number_input(
            "Balance", min_value=0.0, value=50000.0, step=1000.0
        )

        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=50000.0, step=1000.0
        )

        has_cr_card = st.selectbox(
            "Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )

        is_active_member = st.selectbox(
            "Is Active Member",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    st.divider()

    submitted = st.form_submit_button("Predict Churn Risk", width="stretch")


# ============================================================
# PREDICTION
# ============================================================

if submitted:
    # --------------------------------------------------------
    # ENCODE GENDER
    # --------------------------------------------------------
    encoded_gender = gender_encoder.transform([gender])[0]

    # --------------------------------------------------------
    # BASE INPUT ROW
    # --------------------------------------------------------
    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Gender": [encoded_gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
    )

    # --------------------------------------------------------
    # ONE-HOT ENCODE GEOGRAPHY
    # --------------------------------------------------------
    geography_encoded = geography_encoder.transform(
        pd.DataFrame({"Geography": [geography]})
    ).toarray()

    geography_encoded_df = pd.DataFrame(
        geography_encoded,
        columns=geography_encoder.get_feature_names_out(["Geography"]),
    )

    input_data = pd.concat(
        [input_data.reset_index(drop=True),
         geography_encoded_df.reset_index(drop=True)],
        axis=1,
    )

    # --------------------------------------------------------
    # ENFORCE TRAINING-TIME FEATURE ORDER
    # The scaler was fit on a DataFrame, so it remembers the exact
    # column order and names it saw during training. Re-deriving the
    # order from the scaler (instead of trusting the dict order above)
    # protects against silent feature-mismatch bugs — the kind that
    # don't crash, they just quietly produce a wrong prediction.
    # --------------------------------------------------------
    expected_columns = list(scaler.feature_names_in_)
    input_data = input_data[expected_columns]

    # --------------------------------------------------------
    # SCALE + PREDICT
    # --------------------------------------------------------
    input_scaled = scaler.transform(input_data)
    prediction = model(input_scaled, training=False).numpy()
    churn_probability = float(prediction[0][0])

    # --------------------------------------------------------
    # RISK LEVEL
    # -------------------------------------------------------- 
    if churn_probability < 0.30:
        risk_level = "LOW RISK"
        recommendation = (
            "Customer currently shows relatively low predicted churn risk. "
            "Continue normal engagement."
        )
    elif churn_probability < 0.60:
        risk_level = "MODERATE RISK"
        recommendation = (
            "Customer shows moderate churn risk. Consider proactive "
            "engagement and an account review."
        )
    else:
        risk_level = "HIGH RISK"
        recommendation = (
            "Customer shows elevated churn risk. Consider a targeted "
            "retention intervention."
        )

    # ========================================================
    # RESULT SECTION
    # ========================================================
    st.divider()
    st.subheader("Prediction Result")

    result_col, profile_col = st.columns([1.2, 1])

    with result_col:
        st.metric("Predicted Churn Probability", f"{churn_probability:.1%}")
        st.progress(min(churn_probability, 1.0))
        st.markdown(f"### {risk_level}")

        if churn_probability >= 0.60:
            st.error("Customer is likely to churn.")
        elif churn_probability >= 0.30:
            st.warning("Customer has moderate churn risk.")
        else:
            st.success("Customer is currently at lower churn risk.")

        st.markdown(
            f"""
            <div class="info-card">
                <strong>Recommended action</strong><br><br>
                {recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Note: this score is the model's raw sigmoid output, treated as "
            "a churn probability. It has not been calibrated against actual "
            "observed churn rates."
        )

    with profile_col:
        st.subheader("Customer Snapshot")
        st.write(f"**Geography:** {geography}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Age:** {age}")
        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Balance:** {balance:,.2f}")
        st.write(f"**Estimated Salary:** {estimated_salary:,.2f}")
        st.write(f"**Tenure:** {tenure} years")
        st.write(f"**Products:** {num_of_products}")
        st.write(f"**Has Credit Card:** {'Yes' if has_cr_card else 'No'}")
        st.write(f"**Active Member:** {'Yes' if is_active_member else 'No'}")

    with st.expander("View raw model input (post-preprocessing)"):
        st.dataframe(input_data, width="stretch")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Built with TensorFlow, Scikit-learn, Pandas and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
