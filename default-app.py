import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load the trained model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

st.title("Loan Default Prediction App")

# Collect inputs from user
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.number_input("Months Employed", min_value=0, value=12)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=2)
interest_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)
loan_term = st.number_input("Loan Term (months)", min_value=6, value=36)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.4)

education = st.selectbox("Education", [0, 1, 2, 3])  # replace with real mapping if you had labels
employment_type = st.selectbox("Employment Type", [0, 1, 2, 3])
marital_status = st.selectbox("Marital Status", [0, 1, 2, 3])
has_mortgage = st.selectbox("Has Mortgage", [0, 1])
has_dependents = st.selectbox("Has Dependents", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", [0, 1, 2, 3, 4])
has_cosigner = st.selectbox("Has CoSigner", [0, 1])

if st.button("Predict"):
    # Order of features must match training
    input_dict = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }

    input_data = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"### Prediction: {'Default' if prediction == 1 else 'No Default'}")
    st.write(f"### Probability of Default: {probability:.2f}")
