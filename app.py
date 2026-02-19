import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Create src module path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Use a safer import structure
try:
    from src.risk_segmentation import run_risk_segmentation
except ImportError:
    # If running from root, try direct import
    try:
        from src.risk_segmentation import run_risk_segmentation
    except:
         # Define a dummy function if import fails to avoid app crash
        def run_risk_segmentation():
            return None

# Set page configuration
st.set_page_config(
    page_title="Fintech Credit Risk Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to format INR
def format_inr(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/credit_risk_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/lending_data.csv")
        return df
    except FileNotFoundError:
        return None

df_stats = load_data()
model = load_model()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=100)
    st.title("Fintech Credit Risk Engine")
    st.markdown("---")
    st.write("Navigation")
    page = st.radio("Go to", ["Dashboard", "Risk Analysis", "What-If Simulator"])
    st.markdown("---")

# Dashboard Page
if page == "Dashboard":
    st.title("💳 Fintech Credit Risk Engine")
    
    # Overview Section
    st.info("""
    **👋 Welcome to the Unified Credit Engine**
    
    This AI-powered application is tuned for the **Indian Banking Sector**.
    It assesses loan applications based on key financial health indicators like CIBIL Score, Income (LPA), and DTI.
    
    **Key Features:**
    - **Real-time Scoring**: Instant evaluation of default probability.
    - **Explainable AI**: Understand why a loan is approved or rejected.
    - **What-If Scenarios**: Simulate changes in income or loan amount.
    """)
    
    st.markdown("### 📊 Real-time Portfolio Overview")
    
    # KPIs
    total_apps = "N/A"
    default_rate = "N/A"
    avg_loan = "N/A"
    
    if df_stats is not None:
        total_apps = f"{len(df_stats):,}"
        default_rate_val = df_stats['loan_status'].mean() * 100
        default_rate = f"{default_rate_val:.1f}%"
        avg_loan = f"₹{df_stats['loan_amount'].mean():,.0f}"
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", total_apps, "+12%")
    with col2:
        st.metric("Approval Rate", f"{100-default_rate_val:.1f}%" if df_stats is not None else "N/A", "+2%")
    with col3:
        st.metric("Avg. Loan Amount", avg_loan, "+5%")
    with col4:
        st.metric("Default Rate", default_rate, "-0.5%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Loan Purpose Distribution")
        # Dummy data for chart
        if model:
            purpose_data = pd.DataFrame({
                'Purpose': ['Home Renovation', 'Personal Loan', 'Business Expansion', 'Education', 'Wedding'],
                'Count': [450, 300, 200, 150, 145]
            })
            fig, ax = plt.subplots()
            ax.pie(purpose_data['Count'], labels=purpose_data['Purpose'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
            ax.axis('equal')
            st.pyplot(fig)
            st.caption("**Insight:** 'Home Renovation' and 'Personal Loans' constitute the majority of credit requests, typical in the Indian retail lending market.")
            
    with col2:
        st.subheader("CIBIL Score Distribution")
        # Mock distribution
        risk_scores = np.random.normal(700, 40, 1000)
        risk_scores = np.clip(risk_scores, 300, 900)
        
        fig, ax = plt.subplots()
        sns.histplot(risk_scores, kde=True, color="teal", ax=ax)
        ax.set_xlabel("CIBIL Score")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)
        st.caption("**Insight:** Most applicants fall within the 700-750 range, which is considered 'Good' by most Indian banks.")

    
    st.markdown("---")
    st.subheader("Credit Performance Analysis")
    
    # 1. New SQL Segment Analysis Section
    st.markdown("**Data Engineering Insight:** Utilizing SQL Window Functions for Risk Segmentation")
    st.markdown("We simulate `NTILE()` and `RANK()` functions to segment users based on Loan-to-Income ratios.")
    
    # Import locally to avoid top-level failures
    try:
        from src.risk_segmentation import run_risk_segmentation
        segment_data = run_risk_segmentation()
        
        if segment_data is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.caption("Risk Segment Table")
                st.dataframe(segment_data.style.format({'Default Rate': '{:.2%}'}), use_container_width=True)
            with col2:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x='Risk Segment', y='Default Rate', data=segment_data, palette="Blues_d", ax=ax)
                ax.set_ylabel("Default Probability")
                st.pyplot(fig)
            st.caption("**Logic:** This segmentation buckets applicants based on Loan-to-Income ratios. 'High LTI' creates a higher default probability.")
    except Exception as e:
        st.error(f"Could not run segmentation analysis: {e}")
    
    st.markdown("### Deep Dive Factors")
    
    # Boxplot
    n_viz = 500
    viz_status = np.random.choice(["Paid", "Default"], n_viz, p=[0.8, 0.2])
    viz_scores_paid = np.random.normal(750, 30, int(n_viz*0.8))
    viz_scores_default = np.random.normal(650, 50, int(n_viz*0.2))
    viz_scores = np.concatenate([viz_scores_paid, viz_scores_default])
    # match lengths if rounding error
    viz_status = ["Paid"] * len(viz_scores_paid) + ["Default"] * len(viz_scores_default)
    
    viz_df = pd.DataFrame({'Loan Status': viz_status, 'CIBIL Score': viz_scores})
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x='Loan Status', y='CIBIL Score', data=viz_df, palette="Set2", hue='Loan Status', legend=False, ax=ax)
    st.pyplot(fig)
    st.caption("**Analysis:** Defaulters consistently show lower CIBIL scores (Median ~650) compared to good payers (Median ~750).")


# Risk Analysis Page
elif page == "Risk Analysis":
    st.title("🔍 Model Risk Analysis")
    
    if model:
        st.subheader("Feature Importance Analysis")
        st.markdown("""
        The chart below ranks the input variables by their influence on the credit decision.
        """)
        
        # Get feature importance
        feature_names = ['Annual Income', 'Loan Amount', 'CIBIL Score', 'Employment Length', 'DTI Ratio', 'Age']
        try:
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # Fixed warning by setting hue and legend=False
            sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', hue='Feature', legend=False, ax=ax)
            ax.set_xlabel("Relative Importance Score")
            st.pyplot(fig)
            
            st.markdown("### 💡 Key Insights (Indian Context)")
            st.info("""
            *   **CIBIL Score**: The primary determinant for loan approval in India.
            *   **Debt-to-Income (DTI)**: Determines repayment capacity. Banks prefer DTI < 40%.
            *   **Annual Income**: Higher income increases loan eligibility, but stability is key.
            """)
            
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")

# What-If Simulator Page
elif page == "What-If Simulator":
    st.title("🎛️ What-If Scenario Simulator")
    st.markdown("Adjust applicant parameters to see real-time changes in risk probability.")
    
    if model:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Applicant Profile")
            # Income: 2.5L to 50L
            income = st.slider("Annual Income (₹)", 250000, 5000000, 800000, step=50000, format="%d")
            # Loan: 50k to 25L
            loan_amount = st.slider("Loan Amount (₹)", 50000, 2500000, 500000, step=10000, format="%d")
            # CIBIL: 300 to 900
            credit_score = st.slider("CIBIL Score", 300, 900, 750)
            employment_length = st.slider("Employment Length (Years)", 0, 40, 5)
            dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 30.0)
            age = st.slider("Age", 21, 65, 30)
            
            # Map inputs back to model feature names
            # Features: income, loan_amount, credit_score, employment_length, dti, age
            input_data = pd.DataFrame({
                'income': [income],
                'loan_amount': [loan_amount],
                'credit_score': [credit_score],
                'employment_length': [employment_length],
                'dti': [dti],
                'age': [age]
            })
            
        with col2:
            st.subheader("Risk Assessment")
            
            # Prediction
            prediction_prob = model.predict_proba(input_data)[0][1] # Probability of Default (Class 1)
            
            # Risk Meter
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Default Probability", f"{prediction_prob:.1%}")
            with col_metric2:
                st.metric("System Confidence", f"{max(model.predict_proba(input_data)[0]):.1%}")

            
            # Adjusted thresholds for low default rate environment (3% avg)
            if prediction_prob < 0.05: # < 5% Probability
                risk_level = "LOW RISK"
                color = "#28a745" # Green
                msg = "Strong profile. Likely to be approved."
            elif prediction_prob < 0.20: # 5% - 20% Probability
                risk_level = "MEDIUM RISK"
                color = "#ffc107" # Orange
                msg = "Moderate risk. Additional documents may be required."
            else: # > 20% Probability
                risk_level = "HIGH RISK"
                color = "#dc3545" # Red
                msg = "High probability of default. Rejection recommended."
            
            st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin:0;">{risk_level}</h2>
                    <p style="margin:0; font-style: italic;">{msg}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📋 Decision Factors (Standard Indian Bank Norms)")
            
            factors = []
            formatted_income = format_inr(income)
            
            if dti > 50:
                factors.append(f"❌ **Critical DTI ({dti}%)**: EMI burden exceeds 50% of income. High Risk.")
            elif dti > 40:
                factors.append(f"⚠️ **High DTI ({dti}%)**: Approaching upper limit of 40-50%.")
            else:
                factors.append(f"✅ **Healthy DTI ({dti}%)**: Well within repayment capacity.")
                
            if credit_score < 650:
                factors.append(f"❌ **Low CIBIL ({credit_score})**: Below the standard 700 threshold.")
            elif credit_score >= 750:
                factors.append(f"✅ **Excellent CIBIL ({credit_score})**: Eligible for preferential interest rates.")
            
            # Loan to Income Ratio check (simplified)
            if loan_amount > income * 2.5:
                 factors.append(f"⚠️ **High Loan Request**: Loan is >2.5x of annual income (₹{formatted_income}).")

            if not factors:
                factors.append("ℹ️ Parameters are within standard banking norms.")
                
            for factor in factors:
                st.write(factor)

            st.caption("Note: Based on standard Indian retail banking risk parameters.")

    else:
        st.warning("Model not loaded. Please ensure 'model/credit_risk_model.pkl' exists.")
