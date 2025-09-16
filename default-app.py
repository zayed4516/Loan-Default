import streamlit as st
import pandas as pd
import pickle

# تحميل النموذج
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📊 Loan Default Prediction App")
st.write("Fill in the applicant details below to predict loan default.")

# إدخال البيانات
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income ($)", min_value=0, max_value=1000000, value=50000)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=10000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=12)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=50, value=5)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=60)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=5.0, value=0.3)

education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
has_mortgage = st.selectbox("Has Mortgage", [0, 1])
has_dependents = st.selectbox("Has Dependents", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Home", "Car", "Other"])
has_cosigner = st.selectbox("Has Co-Signer", [0, 1])

# تجهيز الداتا للإدخال
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "LoanAmount": [loan_amount],
    "CreditScore": [credit_score],
    "MonthsEmployed": [months_employed],
    "NumCreditLines": [num_credit_lines],
    "InterestRate": [interest_rate],
    "LoanTerm": [loan_term],
    "DTIRatio": [dti_ratio],
    "Education": [education],
    "EmploymentType": [employment_type],
    "MaritalStatus": [marital_status],
    "HasMortgage": [has_mortgage],
    "HasDependents": [has_dependents],
    "LoanPurpose": [loan_purpose],
    "HasCoSigner": [has_cosigner],
})

# التنبؤ
if st.button("Predict Default"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ This loan is likely to **default**.")
    else:
        st.success("✅ This loan is likely to be **paid back**.")

