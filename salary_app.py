from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="ANN Salary Estimator",
    page_icon="💼",
    layout="centered",
)


# ------------------------------------------------------------
# Resolve files relative to this app.py file.
# This keeps the app working when launched from the repository root.
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "salary_regression_model.keras"
GENDER_ENCODER_PATH = BASE_DIR / "label_encoder_gender.pkl"
GEO_ENCODER_PATH = BASE_DIR / "one_hot_encoder_geo.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


@st.cache_resource
def load_artifacts():
    """Load the trained model and the exact preprocessing objects used in training."""
    model = load_model(MODEL_PATH, compile=False)

    with open(GENDER_ENCODER_PATH, "rb") as file:
        gender_encoder = pickle.load(file)

    with open(GEO_ENCODER_PATH, "rb") as file:
        geography_encoder = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return model, gender_encoder, geography_encoder, scaler


# ------------------------------------------------------------
# Page header
# ------------------------------------------------------------
st.title("ANN Salary Estimator")
st.caption(
    "Bank Customer Estimated Salary Regression using an Artificial Neural Network"
)

st.markdown(
    """
    This application uses a trained TensorFlow/Keras ANN to estimate the
    `EstimatedSalary` value from a customer's profile.

    **This is an educational regression experiment.**
    The prediction should not be treated as a guaranteed real-world salary,
    compensation, or financial estimate.
    """
)

# ------------------------------------------------------------
# Load the trained artifacts
# ------------------------------------------------------------
try:
    model, gender_encoder, geography_encoder, scaler = load_artifacts()
except FileNotFoundError as exc:
    st.error(
        f"Required model/preprocessing file was not found: {exc.filename}"
    )
    st.info(
        "Make sure salary_regression_model.keras, "
        "label_encoder_gender.pkl, one_hot_encoder_geo.pkl, "
        "and scaler.pkl are in the same folder as salary_app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model or preprocessing artifacts: {exc}")
    st.stop()


# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650,
        step=1,
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
    )

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=75000.0,
        step=1000.0,
    )

with col2:
    geography = st.selectbox(
        "Geography",
        options=["France", "Germany", "Spain"],
    )

    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"],
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )

    has_credit_card = st.selectbox(
        "Has Credit Card",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

is_active_member = st.selectbox(
    "Active Member",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("Estimate Salary", type="primary", use_container_width=True):
    try:
        # Encode Gender using the fitted training encoder.
        gender_encoded = gender_encoder.transform([gender])[0]

        # One-hot encode Geography using the fitted training encoder.
        geo_encoded = geography_encoder.transform(
            [[geography]]
        ).toarray()[0]

        # Reconstruct the same feature order used during training.
        input_data = pd.DataFrame(
            [[
                credit_score,
                gender_encoded,
                age,
                tenure,
                balance,
                num_products,
                has_credit_card,
                is_active_member,
                *geo_encoded,
            ]],
            columns=[
                "CreditScore",
                "Gender",
                "Age",
                "Tenure",
                "Balance",
                "NumOfProducts",
                "HasCrCard",
                "IsActiveMember",
                "Geography_France",
                "Geography_Germany",
                "Geography_Spain",
            ],
        )

        # Apply the exact scaler fitted during training.
        input_scaled = scaler.transform(input_data)

        # Generate the ANN prediction.
        prediction = float(
            np.asarray(
                model.predict(input_scaled, verbose=0)
            ).ravel()[0]
        )

        st.divider()
        st.metric(
            label="Estimated Salary",
            value=f"{prediction:,.2f}",
        )

        st.caption(
            "Prediction generated by the trained ANN using the saved preprocessing pipeline."
        )

    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption(
    "Educational project • TensorFlow/Keras • Scikit-learn • Streamlit"
)
