import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open("catboost_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'catboost_model.pkl' is in the correct directory.")
        return None

model = load_model()

# Professional CSS styling
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1x8cf1d {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: 3px solid #3498db;
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Sidebar text styling */
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stNumberInput label,
    .css-1d391kg .stRadio label,
    .css-1d391kg h2 {
        color: #ecf0f1 !important;
        font-weight: 500;
        font-size: 14px;
    }
    
    /* Input field styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        border: 1px solid #bdc3c7;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #3498db, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.6;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 12px;
        height: 60px;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.6);
        background: linear-gradient(45deg, #2980b9, #1f4e79);
    }
    
    /* Prediction result cards */
    .prediction-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    
    .medium-risk {
        background: linear-gradient(135deg, #ffd43b, #fd7e14);
        color: white;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(116, 185, 255, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Risk gauge styling */
    .risk-gauge {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    /* Professional form sections */
    .form-section {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3498db;
    }
    
    .form-section h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<h1 class="main-header">🏦 Loan Default Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML-powered credit risk evaluation system for informed lending decisions</p>', unsafe_allow_html=True)

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top: 0; color: white;">📊 How it works</h3>
        <p style="margin-bottom: 0; color: rgba(255,255,255,0.9);">
        Our advanced CatBoost model analyzes multiple financial and demographic factors 
        to provide accurate default risk predictions with confidence scores.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("## 📋 Application Details")
    
    # Personal Information Section
    st.markdown('<div class="form-section"><h3>👤 Personal Information</h3>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Applicant's age in years")
    
    education_options = ["High School", "Bachelor", "Master", "PhD"]
    education = st.selectbox("Education Level", education_options, index=1, help="Highest education completed")
    
    marital_status_options = ["Single", "Married", "Divorced"]
    marital_status = st.selectbox("Marital Status", marital_status_options, help="Current marital status")
    
    has_dependents = st.radio("Has Dependents", ["No", "Yes"], horizontal=True, help="Financial dependents")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial Information Section
    st.markdown('<div class="form-section"><h3>💰 Financial Information</h3>', unsafe_allow_html=True)
    income = st.number_input("Annual Income ($)", min_value=0, value=65000, step=1000, 
                            help="Total annual income before taxes")
    
    employment_type_options = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    employment_type = st.selectbox("Employment Type", employment_type_options, help="Current employment status")
    
    months_employed = st.number_input("Months Employed", min_value=0, value=24, 
                                     help="Total months in current employment")
    
    dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01,
                         help="Monthly debt payments / Monthly income")
    
    has_mortgage = st.radio("Has Mortgage", ["No", "Yes"], horizontal=True, help="Current mortgage obligations")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Loan Information Section
    st.markdown('<div class="form-section"><h3>🏦 Loan Information</h3>', unsafe_allow_html=True)
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=25000, step=1000,
                                 help="Requested loan amount")
    
    interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=12.5, step=0.1,
                                   help="Annual interest rate")
    
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=2,
                            help="Loan repayment period")
    
    loan_purpose_options = ["Personal", "Business", "Education", "Medical"]
    loan_purpose = st.selectbox("Loan Purpose", loan_purpose_options, help="Primary use of loan funds")
    
    has_cosigner = st.radio("Has Co-Signer", ["No", "Yes"], horizontal=True, help="Additional guarantor")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Credit Information Section
    st.markdown('<div class="form-section"><h3>📈 Credit Information</h3>', unsafe_allow_html=True)
    credit_score_options = {
        "Poor (300-579)": 0,
        "Fair (580-669)": 1,
        "Good (670-739)": 2,
        "Very Good (740-799)": 3,
        "Excellent (800-850)": 4
    }
    credit_score_display = st.selectbox("Credit Score Range", list(credit_score_options.keys()), index=2,
                                       help="Current credit score category")
    credit_score_value = credit_score_options[credit_score_display]
    
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=20, value=3,
                                      help="Total number of active credit accounts")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if model is not None:
    # Create input mapping
    education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    employment_map = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
    loan_purpose_map = {"Personal": 0, "Business": 1, "Education": 2, "Medical": 3}
    yes_no_map = {"Yes": 1, "No": 0}

    # Prepare input data
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
        "interest_value": loan_amount * interest_rate / 100
    }])

    # Create columns for the main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Display key metrics
        st.markdown("## 📊 Application Summary")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Loan Amount", f"${loan_amount:,}", help="Requested loan amount")
        
        with metrics_col2:
            st.metric("Monthly Payment", f"${loan_amount * (interest_rate/100/12) / (1 - (1 + interest_rate/100/12)**(-loan_term)):,.0f}")
        
        with metrics_col3:
            st.metric("DTI Ratio", f"{dti_ratio:.1%}", help="Debt-to-Income ratio")
        
        with metrics_col4:
            st.metric("Credit Score", credit_score_display.split('(')[0].strip())

        # Prediction button and results
        st.markdown("## 🔮 Risk Assessment")
        
        if st.button("🚀 Analyze Default Risk", use_container_width=True):
            with st.spinner("Analyzing creditworthiness..."):
                try:
                    prediction = model.predict(input_data)[0]
                    proba = model.predict_proba(input_data)[0][1]
                    
                    # Create risk gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = proba * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Default Risk %"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        font={'color': "darkblue", 'family': "Inter"},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk assessment result
                    if proba < 0.3:
                        risk_level = "LOW RISK"
                        risk_class = "low-risk"
                        risk_emoji = "✅"
                        recommendation = "Recommended for approval with standard terms."
                    elif proba < 0.6:
                        risk_level = "MEDIUM RISK"
                        risk_class = "medium-risk"
                        risk_emoji = "⚠️"
                        recommendation = "Consider approval with enhanced monitoring or adjusted terms."
                    else:
                        risk_level = "HIGH RISK"
                        risk_class = "high-risk"
                        risk_emoji = "🚨"
                        recommendation = "Requires careful review. Consider additional collateral or co-signer."
                    
                    st.markdown(f"""
                    <div class="prediction-card {risk_class}">
                        <h2 style="margin-top: 0;">{risk_emoji} {risk_level}</h2>
                        <h3>Default Probability: {proba:.1%}</h3>
                        <p style="margin-bottom: 0; font-size: 1rem; opacity: 0.9;">
                            {recommendation}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("### 📈 Risk Factors Analysis")
                    
                    # Create risk factors visualization
                    factors = {
                        'Credit Score': (5 - credit_score_value) * 20,  # Lower score = higher risk
                        'DTI Ratio': dti_ratio * 100,
                        'Employment Stability': max(0, (12 - months_employed) * 5),
                        'Loan Amount vs Income': (loan_amount / income) * 100
                    }
                    
                    factors_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Risk Score'])
                    
                    fig2 = px.bar(factors_df, x='Factor', y='Risk Score', 
                                 color='Risk Score',
                                 color_continuous_scale=['green', 'yellow', 'red'])
                    fig2.update_layout(
                        title="Individual Risk Factor Contribution",
                        xaxis_title="Risk Factors",
                        yaxis_title="Risk Score (0-100)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {str(e)}")
                    st.info("Please check that all required fields are filled correctly.")

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Loan Default Risk Assessment System</strong></p>
    <p>Powered by Advanced Machine Learning • Last Updated: {}</p>
    <p><small>⚠️ This tool provides risk assessment guidance only. Final lending decisions should consider additional factors and comply with applicable regulations.</small></p>
</div>
""".format(datetime.now().strftime("%B %Y")), unsafe_allow_html=True)
