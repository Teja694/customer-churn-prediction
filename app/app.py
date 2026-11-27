
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# Load the trained model
# --------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "xgboost_churn_model.pkl"
)

model = joblib.load(MODEL_PATH)

# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether "
    "the customer is likely to leave the service."
)

st.divider()

# --------------------------------
# Customer information form
# --------------------------------

with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1]
        )

        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.divider()

    # --------------------------------
    # Additional services
    # --------------------------------

    st.subheader("Additional Services")

    col3, col4 = st.columns(2)

    with col3:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col4:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    st.divider()

    # --------------------------------
    # Billing information
    # --------------------------------

    st.subheader("Billing Information")

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=1.0
    )

    st.caption(
        "Enter the customer's billing amounts in the "
        "same currency and units as the training dataset."
    )

    submit = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# --------------------------------
# Prediction
# --------------------------------

if submit:

    # Check billing information
    if tenure == 0 and total_charges != 0:
        st.warning(
            "For a customer with zero tenure, "
            "Total Charges should normally be zero."
        )

    else:

        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        customer_df = pd.DataFrame([customer_data])

        # Get prediction and probability
        prediction = model.predict(customer_df)[0]

        churn_probability = model.predict_proba(
            customer_df
        )[0][1]

        # --------------------------------
        # Display prediction result
        # --------------------------------

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                "⚠️ This customer is likely to churn."
            )
        else:
            st.success(
                "✅ This customer is unlikely to churn."
            )

        # Churn probability
        st.write("### Churn Probability")

        st.progress(float(churn_probability))

        col5, col6 = st.columns(2)

        with col5:
            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with col6:
            st.metric(
                "Classification Threshold",
                "50%"
            )

        # Explanation
        st.caption(
            "The model predicts churn when the estimated "
            "churn probability is 50% or higher."
        )

        st.info(
            "This prediction is based on patterns learned "
            "from historical customer data. It is an estimate, "
            "not a guarantee of future customer behaviour."
        )

# --------------------------------
# Footer
# --------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
