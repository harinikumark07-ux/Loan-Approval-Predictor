import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('log_reg_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Loan Approval Prediction Dashboard")
st.title("Loan Approval Prediction Dashboard")
st.write("This dashboard predicts the probability of loan approval based on applicant financial and credit details using logistic regression.")
# User inputs
gender = st.selectbox("Applicant gender", ["female", "male"])
age = st.number_input("Age", min_value=18, max_value=80, value=30)
education = st.selectbox("Education level", ["High School", "Graduate", "Post-Graduate", "Other"])
income = st.number_input("Annual income", min_value=0, max_value=1000000, value=50000)
emp_exp = st.number_input("Employment experience (years)", min_value=0, max_value=50, value=5)
home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Loan amount", min_value=0, max_value=1000000, value=10000)
loan_intent = st.selectbox("Loan purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_int_rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=50.0, value=12.0)
loan_percent_income = st.number_input("Installment as % of income (0-1)", min_value=0.0, max_value=1.0, value=0.2)
cred_hist_len = st.number_input("Credit history length (years)", min_value=0.0, max_value=50.0, value=5.0)
credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=650)
prev_default = st.selectbox("Previous loan default on file", ["No", "Yes"])

# Build one-row DataFrame
row = {
    "person_age": age,
    "person_gender": gender,
    "person_education": education,
    "person_income": income,
    "person_emp_exp": emp_exp,
    "person_home_ownership": home_ownership,
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cred_hist_len,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": "Y" if prev_default == "Yes" else "N",
}

raw_df = pd.DataFrame([row])

# Same encoding as in training
cat_cols = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]
encoded_df = pd.get_dummies(raw_df, columns=cat_cols, drop_first=True)

# Match columns used in model (missing ones become 0)
encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

if st.button("Check Loan Approval"):
    probability = model.predict_proba(encoded_df)[0, 1]

    st.write(f"Predicted loan approval probability: **{probability:.2%}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Loan Approval Probability", f"{probability * 100:.2f}%")
    with col2:
        st.metric(
            "Risk Level",
            "Low" if probability >= 0.7 else "Medium" if probability >= 0.4 else "High"
        )
    with col3:
        st.metric("Model Type", "Logistic Regression")

    if probability >= 0.7:
        st.success("High probability of loan approval.")
    elif probability >= 0.4:
        st.warning("Moderate probability of loan approval.")
    else:
        st.error("Low probability of loan approval.")

    st.subheader("Key insight")
    if probability >= 0.7:
        st.write("This applicant shows a strong chance of getting loan approval based on the entered business and credit details.")
    elif probability >= 0.4:
        st.write("This applicant shows a moderate chance of loan approval access. Improving credit profile may increase access.")
    else:
        st.write("This applicant shows a low chance of loan approval access. Risk-related factors may be limiting access to credit.")
