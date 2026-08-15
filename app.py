import streamlit as st
import joblib
import pandas as pd

# Load the model once and cache it
@st.cache_resource
def load_model():
    try:
        model = joblib.load("outputs/model.joblib")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is in the 'outputs/model.joblib' directory.")
        return None

model = load_model()

if model is not None:
    st.title("Bank Term Deposit Lead Scoring Tool")
    st.write("This tool scores whether a customer will subscribe to a term deposit.")

    # Define the form fields
    with st.form(key='lead_scoring_form'):
        age = st.number_input("Age", min_value=18, max_value=95, value=30, step=1)
        campaign = st.number_input("Campaign", min_value=1, max_value=50, value=5, step=1)
        pdays = st.number_input("Days since last contact", min_value=0, max_value=999, value=0, step=1)
        previous = st.number_input("Previous contacts", min_value=0, max_value=10, value=0, step=1)
        emp_var_rate = st.number_input("Employment variation rate", min_value=-3.4, max_value=1.4, value=0.0, step=0.1)
        cons_price_idx = st.number_input("Consumer price index", min_value=92.0, max_value=95.0, value=94.0, step=0.1)
        cons_conf_idx = st.number_input("Consumer confidence index", min_value=-50.0, max_value=-26.0, value=-30.0, step=0.1)
        euribor3m = st.number_input("EURIBOR 3 month rate", min_value=0.6, max_value=5.0, value=2.0, step=0.1)
        nr_employed = st.number_input("Number of employees", min_value=4963, max_value=5228, value=5000, step=1)

        job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital status", ["divorced", "married", "single", "unknown"])
        education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"])
        default = st.selectbox("Default", ["no", "yes", "unknown"])
        housing = st.selectbox("Housing", ["no", "yes", "unknown"])
        loan = st.selectbox("Loan", ["no", "yes", "unknown"])
        contact = st.selectbox("Contact method", ["cellular", "telephone"])
        month = st.selectbox("Month", ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        day_of_week = st.selectbox("Day of week", ["mon", "tue", "wed", "thu", "fri"])
        poutcome = st.selectbox("Previous outcome", ["failure", "nonexistent", "success"])

        submitted = st.form_submit_button("Submit")

    if submitted:
        # Create the input DataFrame
        input_data = {
            "age": age,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "poutcome": poutcome
        }

        df = pd.DataFrame([input_data])

        try:
            # Get the prediction probability
            probability = model.predict_proba(df)[:, 1][0]
            prediction = "High" if probability > 0.7 else "Medium" if 0.4 <= probability <= 0.7 else "Low"

            # Display the result
            st.metric("Conversion Probability", f"{probability * 100:.2f}%", prediction)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")