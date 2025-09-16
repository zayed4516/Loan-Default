import streamlit as st
import pandas as pd
import pickle

# تحميل الموديل
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== Streamlit UI =====
st.set_page_config(page_title="Loan Default Prediction", page_icon="💳", layout="wide")

# ===== CSS لتعديل الخلفية والـ sidebar =====
st.markdown(
    """
    <style>
    /* خلفية سوداء للصفحة */
    .stApp {
        background-color: #000000;
        color: white;
    }

    /* خلفية sidebar سوداء */
    .css-1d391kg {  /* ممكن يختلف حسب نسخة Streamlit */
        background-color: #111111;
        color: white;
        width: 350px;
    }

    /* تكبير حجم النصوص داخل sidebar */
    .css-1d391kg label, .css-1d391kg div {
        color: white !important;
    }

    /* تصميم أزرار Prediction */
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }

    /* تصميم الصندوق Prediction */
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        font-size: 18px;
    }

    .high-risk {
        background-color: #FDEDEC;
        border: 1px solid #E74C3C;
        color: #C0392B;
    }

    .low-risk {
        background-color: #E8F8F5;
        border: 1px solid #1ABC9C;
        color: #16A085;
    }

    </style>
    """, unsafe_allow_html=True
)

# ===== عرض الصورة =====
st.image("bank.jpg", use_column_width=True)

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💳 Loan Default Prediction App</h1>", unsafe_allow_html=True)
st.write("### Enter applicant details below to predict the likelihood of loan default.")

# ===== Sidebar Inputs =====
with st.sidebar:
    st.header("Applicant Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income ($)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=20000)
    
    # Credit Score as discrete category
    credit_score_map = {
        "Poor (300-579)": 0,
        "Fair (580-669)": 1,
        "Good (670-739)": 2,
        "Very Good (740-799)": 3,
        "Excellent (800-850)": 4
    }
    credit_score = st.selectbox("Credit Score", list(credit_score_map.keys()))
    credit_score_value = credit_score_map[credit_score]

    months_employed = st.number_input("Months Employed", min_value=0, value=4)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=1)

    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Education", "Medical"])

    has_mortgage = st.radio("Has Mortgage", ["Yes", "No"], horizontal=True)
    has_dependents = st.radio("Has Dependents", ["Yes", "No"], horizontal=True)
    has_cosigner = st.radio("Has Co-Signer", ["Yes", "No"], horizontal=True)

# ===== Mapping for categorical features =====
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
employment_map = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
loan_purpose_map = {"Personal": 0, "Business": 1, "Education": 2, "Medical": 3}
yes_no_map = {"Yes": 1, "No": 0}

# ===== Prepare input data =====
input_data = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score_value,
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

# ===== Prediction Section في منتصف الصفحة =====
st.markdown("<h2 style='text-align: center; color: #2E86C1;'>🔮 Prediction</h2>", unsafe_allow_html=True)
if st.button("Predict Default", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # احتمال التعثر

        if prediction == 1:
            st.markdown(
                f"<div class='prediction-box high-risk'>⚠️ High Risk of Default<br>Probability: <b>{proba:.2%}</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='prediction-box low-risk'>✅ Low Risk of Default<br>Probability: <b>{proba:.2%}</b></div>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
