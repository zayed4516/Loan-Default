import streamlit as st
import pandas as pd
import pickle

# تحميل الموديل
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Loan Default Prediction App")

st.write("Enter applicant details below to predict the likelihood of default.")

# إدخال البيانات من المستخدم
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income ($)", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=20000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.number_input("Months Employed", min_value=0, value=4)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=1)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)
loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)
dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3)

# كاتيجوري يدخلها المستخدم كنص
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Education", "Medical"])

# Boolean يدخلها كنعم/لا
has_mortgage = st.radio("Has Mortgage", ["Yes", "No"])
has_dependents = st.radio("Has Dependents", ["Yes", "No"])
has_cosigner = st.radio("Has Co-Signer", ["Yes", "No"])

# mapping للتحويل
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
employment_map = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
loan_purpose_map = {"Personal": 0, "Business": 1, "Education": 2, "Medical": 3}
yes_no_map = {"Yes": 1, "No": 0}

# تحويل المدخلات
input_data = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "MonthsEmployed": months_employed,
    "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate,
    "LoanTerm": loan_term,
    "DTIRatio": dti_ratio,
    "Education": education_map[education],
    "EmploymentType": employment_map[employment_type],
    "MaritalStatus": marital_map[marital_status],
    "HasMortgage": yes_no_map[has_mortgage],
    "HasDependents": yes_no_map[has_dependents],
    "LoanPurpose": loan_purpose_map[loan_purpose],
    "HasCoSigner": yes_no_map[has_cosigner],
    "interest_value": loan_amount * interest_rate
}])

# prediction
if st.button("Predict Default"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The applicant is likely to default on the loan.")
    else:
        st.success("✅ The applicant is unlikely to default on the loan.")
