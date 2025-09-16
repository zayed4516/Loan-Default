import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ----------------------------
# Load trained CatBoost model
# ----------------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

model = load_model()

# ----------------------------
# Expected Features
# ----------------------------
EXPECTED_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

st.title("💳 Loan Default Prediction App")
st.write("Enter applicant details below to predict the likelihood of default.")

# ----------------------------
# Input Form
# ----------------------------
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        Income = st.number_input("Income ($)", min_value=0, value=50000)
        LoanAmount = st.number_input("Loan Amount ($)", min_value=0, value=20000)
        CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        MonthsEmployed = st.number_input("Months Employed", min_value=0, value=24)
        NumCreditLines = st.number_input("Number of Credit Lines", min_value=0, value=2)
        InterestRate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)
        LoanTerm = st.number_input("Loan Term (months)", min_value=6, value=36)

    with col2:
        DTIRatio = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.3, step=0.01)
        Education = st.selectbox("Education", [0, 1, 2, 3])  # Example encoding
        EmploymentType = st.selectbox("Employment Type", [0, 1, 2, 3])  # Example encoding
        MaritalStatus = st.selectbox("Marital Status", [0, 1])  # Example encoding
        HasMortgage = st.selectbox("Has Mortgage", [0, 1])
        HasDependents = st.selectbox("Has Dependents", [0, 1])
        LoanPurpose = st.selectbox("Loan Purpose", [0, 1, 2, 3, 4])  # Example encoding
        HasCoSigner = st.selectbox("Has Co-Signer", [0, 1])

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    input_data = pd.DataFrame([[
        Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines,
        InterestRate, LoanTerm, DTIRatio, Education, EmploymentType,
        MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner
    ]], columns=EXPECTED_FEATURES)

    # ➕ Add interest_value feature (LoanAmount * InterestRate)
    input_data["interest_value"] = input_data["LoanAmount"] * input_data["InterestRate"]

    # ✅ Ensure same order as training
    input_data = input_data[EXPECTED_FEATURES + ["interest_value"]]

    # Predict
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.success(f"### 🎯 Prediction: {'Default' if prediction==1 else 'No Default'}")
        st.info(f"🔮 Probability of Default: {proba:.2%}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
