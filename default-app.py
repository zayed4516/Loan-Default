import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ================================
# Load trained CatBoost model
# ================================
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# ================================
# Define categorical mappings (same as training)
# ================================
education_map = {
    "High School": 0,
    "Bachelors": 1,
    "Masters": 2,
    "PhD": 3
}

employment_map = {
    "Full-time": 0,
    "Part-time": 1,
    "Self-employed": 2,
    "Unemployed": 3
}

marital_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

loan_purpose_map = {
    "Personal": 0,
    "Education": 1,
    "Medical": 2,
    "Business": 3,
    "Other": 4
}

# ================================
# App Layout
# ================================
st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Default Prediction App")
st.markdown("Fill in the applicant details below to predict the probability of **loan default**.")

# ================================
# User Inputs
# ================================
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income ($)", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.number_input("Months Employed", min_value=0, value=12)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=5)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_term = st.number_input("Loan Term (months)", min_value=1, value=60)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=5.0, value=0.30)

education = st.selectbox("Education", list(education_map.keys()))
employment_type = st.selectbox("Employment Type", list(employment_map.keys()))
marital_status = st.selectbox("Marital Status", list(marital_map.keys()))

has_mortgage = st.selectbox("Has Mortgage", [0, 1])
has_dependents = st.selectbox("Has Dependents", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", list(loan_purpose_map.keys()))
has_cosigner = st.selectbox("Has Co-Signer", [0, 1])

# ================================
# Prepare input data
# ================================
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
    "HasMortgage": has_mortgage,
    "HasDependents": has_dependents,
    "LoanPurpose": loan_purpose_map[loan_purpose],
    "HasCoSigner": has_cosigner
}])

# ================================
# Prediction
# ================================
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of default

    if prediction == 1:
        st.error(f"⚠️ This loan is likely to **default** with a probability of {probability*100:.2f}%")
    else:
        st.success(f"✅ This loan is likely to be **paid back** with a probability of {(1-probability)*100:.2f}%")
