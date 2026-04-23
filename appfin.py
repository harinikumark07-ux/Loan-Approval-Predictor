import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('log_reg_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Loan Approval Prediction Dashboard")

st.title("🏦 Loan Approval Prediction Dashboard")
st.write("This dashboard predicts the probability of loan approval based on applicant financial and credit details using logistic regression.")

# User inputs
gender = st.selectbox("Applicant gender", ["female", "male"])
age = st.number_input("Age", 18, 80, 30)
education = st.selectbox("Education level", ["High School", "Graduate", "Post-Graduate", "Other"])
income = st.number_input("Annual income", 0, 1000000, 50000)
emp_exp = st.number_input("Employment experience (years)", 0, 50, 5)
home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Loan amount", 0, 1000000, 10000)
loan_intent = st.selectbox("Loan purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_int_rate = st.number_input("Interest rate (%)", 0.0, 50.0, 12.0)
loan_percent_income = st.number_input("Installment as % of income (0-1)", 0.0, 1.0, 0.2)
cred_hist_len = st.number_input("Credit history length (years)", 0.0, 50.0, 5.0)
credit_score = st.number_input("Credit score", 300, 850, 650)
prev_default = st.selectbox("Previous loan default on file", ["No", "Yes"])

# Create DataFrame
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

# Encoding
cat_cols = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

encoded_df = pd.get_dummies(raw_df, columns=cat_cols, drop_first=True)
encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Check Loan Approval"):
    probability = model.predict_proba(encoded_df)[0, 1]

    # Title
    st.write(f"### 📊 Loan Approval Probability: {probability:.2%}")

    # Risk Indicator
    if probability >= 0.7:
        st.success("🟢 High Approval Chance")
    elif probability >= 0.4:
        st.warning("🟡 Medium Approval Chance")
    else:
        st.error("🔴 Low Approval Chance")

    # Progress Bar
    st.progress(float(probability))

    # Bar Chart
    chart_data = pd.DataFrame({
        'Category': ['Approval', 'Rejection'],
        'Probability': [probability, 1 - probability]
    })
    st.bar_chart(chart_data.set_index('Category'))

    # Pie-style table
    st.subheader("📊 Approval Distribution")
    pie_data = pd.DataFrame({
        'Category': ['Approval', 'Rejection'],
        'Values': [probability, 1 - probability]
    })
    st.write(pie_data.set_index('Category'))

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Approval Probability", f"{probability*100:.2f}%")
    with col2:
        st.metric("Model", "Logistic Regression")

    # Insights
    st.subheader("📌 Key Factors Affecting Decision")

    if credit_score > 700:
        st.write("✅ High credit score increases approval chances")
    else:
        st.write("⚠️ Low credit score may reduce approval chances")

    if loan_percent_income > 0.4:
        st.write("⚠️ High loan burden compared to income")
    else:
        st.write("✅ Healthy loan-to-income ratio")

    if prev_default == "Yes":
        st.write("⚠️ Previous loan default negatively impacts approval")
    else:
        st.write("✅ No previous defaults improves approval chances")