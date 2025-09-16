import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool

# تحميل النموذج
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# الأعمدة الكاتيجوريكال
categorical_features = [
    "Education",
    "Employment Type",
    "Marital Status",
    "Loan Purpose"
]

st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("🏦 Loan Default Prediction App")
st.write("Fill in the applicant details below to predict loan default.")

# إدخالات المستخدم
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Income ($)", 0, 1000000, 50000)
loan_amount = st.number_input("Loan Amount ($)", 0, 1000000, 10000)
credit_score = st.number_input("Credit Score", 300, 850, 650)
months_employed = st.number_input("Months Employed", 0, 480, 12)
num_credit_lines = st.number_input("Number of Credit Lines", 0, 50, 5)
interest_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 5.0, step=0.1)
loan_term = st.number_input("Loan Term (months)", 1, 480, 60)
dti_ratio = st.number_input("DTI Ratio", 0.0, 5.0, 0.3, step=0.01)

education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
has_mortgage = st.selectbox("Has Mortgage", [0, 1])
has_dependents = st.selectbox("Has Dependents", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Business"])
has_cosigner = st.selectbox("Has Co-Signer", [0, 1])

# عمل DataFrame من القيم المدخلة
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
    "Education": education,
    "Employment Type": employment_type,
    "Marital Status": marital_status,
    "HasMortgage": has_mortgage,
    "HasDependents": has_dependents,
    "LoanPurpose": loan_purpose,
    "HasCoSigner": has_cosigner
}])

# عمل Pool علشان الكاتيجوريكال
pool = Pool(input_data, cat_features=categorical_features)

# زرار التنبؤ
if st.button("Predict"):
    prediction = model.predict(pool)[0]
    proba = model.predict_proba(pool)[0][1]  # احتمال الافتراضي

    if prediction == 1:
        st.error(f"⚠️ The applicant is **likely to default**. Probability: {proba:.2f}")
    else:
        st.success(f"✅ The applicant is **not likely to default**. Probability: {proba:.2f}")
