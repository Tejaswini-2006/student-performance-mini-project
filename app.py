"""
Student Performance Analytics & Grade Prediction System
=======================================================
Academic Machine Learning Mini-Project
Built with Streamlit, Scikit-learn, Pandas, NumPy, and Plotly.
Predicts secondary school student final mathematics grades (G3) out of 20
based on socio-demographic, academic, and behavioral attributes.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Academic SaaS Appearance
CUSTOM_CSS = """
<style>
    /* Global Font & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container background */
    .main {
        background-color: #F8FAFC;
    }
    
    /* Top Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0D9488 100%);
        color: #FFFFFF;
        padding: 2.2rem 2.4rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        line-height: 1.5;
        max-width: 900px;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(13, 148, 136, 0.25);
        color: #5EEAD4;
        border: 1px solid rgba(94, 234, 212, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Professional Card Styling */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.06);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    
    .metric-caption {
        font-size: 0.82rem;
        color: #0D9488;
        font-weight: 500;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Form Group Container */
    .form-group-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    .form-group-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Result Display Cards */
    .result-card-excellent {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #A7F3D0;
        border-radius: 14px;
        padding: 1.8rem;
        color: #065F46;
    }
    
    .result-card-good {
        background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
        border: 1px solid #99F6E4;
        border-radius: 14px;
        padding: 1.8rem;
        color: #115E59;
    }
    
    .result-card-pass {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        border-radius: 14px;
        padding: 1.8rem;
        color: #1E40AF;
    }
    
    .result-card-risk {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #FECACA;
        border-radius: 14px;
        padding: 1.8rem;
        color: #991B1B;
    }
    
    /* Footer Styling */
    .app-footer {
        margin-top: 3.5rem;
        padding: 2rem 1rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: 0.88rem;
    }
    
    .footer-badge {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    /* Clean Sidebar */
    .css-1d391kg {
        background-color: #F8FAFC;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA AND MODEL LOADING UTILITIES
# -----------------------------------------------------------------------------
MODEL_FILE = "student_performance_random_forest.pkl"
LINEAR_MODEL_FILE = "student_performance_model.pkl"
DATASET_FILE = "student-mat.csv"
IMPORTANCE_FILE = "student_feature_importance.csv"
PREDICTION_RESULTS_FILE = "student_prediction_results.csv"

# Historical Verified Evaluation Metrics (From experimental notebook & verified validation)
VERIFIED_METRICS = {
    "rf_mae": 2.998,
    "rf_rmse": 3.795,
    "rf_r2": 0.298,
    "rf_cv_r2": 0.273,
    "lr_mae": 3.395,
    "lr_rmse": 4.196,
    "lr_r2": 0.142,
    "tuned_rf_mae": 3.108,
    "tuned_rf_rmse": 3.910,
    "tuned_rf_r2": 0.254,
}

@st.cache_resource(show_spinner="Loading trained machine learning pipeline...")
def load_ml_model(filepath=MODEL_FILE):
    """Loads the serialized Scikit-learn Pipeline containing preprocessor and model."""
    if not os.path.exists(filepath):
        return None, f"Model file '{filepath}' was not found in the workspace."
    try:
        model = joblib.load(filepath)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_data(show_spinner="Loading student dataset...")
def load_dataset(filepath=DATASET_FILE):
    """Loads the UCI Mathematics dataset."""
    if not os.path.exists(filepath):
        return None, f"Dataset file '{filepath}' was not found."
    try:
        # Check delimiter (semicolon vs comma)
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(filepath, sep=sep)
        return df, None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

@st.cache_data(show_spinner="Loading test evaluation results...")
def load_test_results(filepath=PREDICTION_RESULTS_FILE):
    """Loads the verified test set predictions for interactive visual error analysis."""
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner="Loading feature importance rankings...")
def load_feature_importance(filepath=IMPORTANCE_FILE):
    """Loads the extracted Random Forest feature importances."""
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception:
            return None
    return None

# Load resources
rf_model, model_error = load_ml_model(MODEL_FILE)
df_raw, data_error = load_dataset(DATASET_FILE)
df_test_results = load_test_results()
df_importance = load_feature_importance()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Student Analytics")
    st.markdown("**Academic Performance Prediction System**")
    
    # Model Status Indicator
    if rf_model is not None:
        st.success("🟢 ML Pipeline Active: Ready")
    else:
        st.error(f"🔴 Pipeline Error: {model_error}")
        
    st.markdown("---")
    
    nav_selection = st.radio(
        "Navigation Menu",
        [
            "🏠 Overview & Home",
            "🎯 Student Prediction",
            "📊 Analytics Dashboard",
            "📖 About Project",
        ],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("#### 📌 Key Model Specifications")
    st.markdown(
        """
        - **Algorithm**: Random Forest Regressor
        - **Target**: Final Math Grade ($G3 \\in [0, 20]$)
        - **Predictors**: 30 Demographic & Behavioral
        - **Excluded**: $G1, G2$ (deliberate)
        - **Test MAE**: 2.998 grade points
        - **Test $R^2$ Score**: 0.298
        """
    )
    st.markdown("---")
    st.caption("Developed by Tejaswini Rakhunde • IT Dept")

# -----------------------------------------------------------------------------
# 4. VIEW: 🏠 OVERVIEW & HOME
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Overview & Home":
    # Hero Section
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">Academic Machine Learning Mini-Project</div>
            <div class="hero-title">Student Performance Analytics</div>
            <div class="hero-subtitle">
                An intelligent predictive analytics platform designed to evaluate socio-demographic, 
                study habit, and environmental indicators to estimate final secondary-school mathematics 
                achievement without relying on mid-term exam scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Summary KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Dataset Size</div>
                <div class="metric-value">{len(df_raw) if df_raw is not None else 395}</div>
                <div class="metric-caption">Student Instances (UCI Math)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Input Dimensions</div>
                <div class="metric-value">30</div>
                <div class="metric-caption">Demographic & Lifestyle Attributes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test MAE</div>
                <div class="metric-value">{VERIFIED_METRICS['rf_mae']}</div>
                <div class="metric-caption">Mean Absolute Error (Scale 0-20)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Model $R^2$ Score</div>
                <div class="metric-value">{VERIFIED_METRICS['rf_r2']}</div>
                <div class="metric-caption">Variance Explained (Non-Exam Baseline)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Pillars
    st.markdown('<div class="section-header">💡 Project Pillars & Capabilities</div>', unsafe_allow_html=True)
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    with col_feat1:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🎯</div>
                <h4 style="color: #0F172A; margin-bottom: 0.5rem;">Early Risk Detection</h4>
                <p style="color: #475569; font-size: 0.9rem; line-height: 1.5;">
                    By intentionally excluding prior exam results ($G1$ and $G2$), the system predicts outcomes 
                    at the beginning of the school term, giving educators early opportunity to plan proactive academic interventions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_feat2:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📊</div>
                <h4 style="color: #0F172A; margin-bottom: 0.5rem;">Multidimensional Insights</h4>
                <p style="color: #475569; font-size: 0.9rem; line-height: 1.5;">
                    Incorporates diverse behavioral determinants: parental education, weekly study hours, 
                    school absences, commute duration, and alcohol consumption habits to model student success holistically.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_feat3:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🔬</div>
                <h4 style="color: #0F172A; margin-bottom: 0.5rem;">Transparent ML Evaluation</h4>
                <p style="color: #475569; font-size: 0.9rem; line-height: 1.5;">
                    Comprehensive diagnostic suite featuring 5-fold cross-validation, hyperparameter comparison, 
                    Gini feature importance rankings, and residual analysis across distinct grade tiers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Navigation Call-To-Action
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.6rem; text-align: center;">
            <h3 style="color: #0F172A; margin-bottom: 0.4rem;">Ready to evaluate a student profile?</h3>
            <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 1.2rem;">
                Select from prepared benchmark profiles or configure all 30 individual parameters to test the live Random Forest pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 5. VIEW: 🎯 STUDENT PREDICTION FORM
# -----------------------------------------------------------------------------
elif nav_selection == "🎯 Student Prediction":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <div class="hero-badge">Interactive Inference Engine</div>
            <div class="hero-title" style="font-size: 1.8rem;">Student Grade Prediction Form</div>
            <div class="hero-subtitle">
                Enter student background parameters across 6 structured categories. 
                The serialized Random Forest pipeline automatically handles categorical one-hot encoding 
                and outputs an estimated final mathematics grade ($G3$) out of 20.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Preset Benchmark Profiles for Viva & Quick Demos
    st.markdown("#### ⚡ Quick Preset Profiles (For viva demonstrations & testing)")
    preset_cols = st.columns([1, 1, 1, 1])
    
    # Initialize session state for inputs if not present
    if "preset_chosen" not in st.session_state:
        st.session_state.preset_chosen = "Default"
        
    with preset_cols[0]:
        if st.button("🎓 High Achiever", use_container_width=True):
            st.session_state.preset_chosen = "High Achiever"
            st.rerun()
    with preset_cols[1]:
        if st.button("📚 Average Student", use_container_width=True):
            st.session_state.preset_chosen = "Average"
            st.rerun()
    with preset_cols[2]:
        if st.button("⚠️ At-Risk Student", use_container_width=True):
            st.session_state.preset_chosen = "At-Risk"
            st.rerun()
    with preset_cols[3]:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.preset_chosen = "Default"
            st.rerun()

    preset = st.session_state.preset_chosen
    
    # Configure preset dictionary values
    defaults = {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 3, "Fedu": 3, "Mjob": "services", "Fjob": "other", "reason": "course", "guardian": "mother",
        "traveltime": 1, "studytime": 2, "failures": 0, "higher": "yes",
        "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "yes", "nursery": "yes", "internet": "yes",
        "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4, "romantic": "no",
        "absences": 4
    }
    
    if preset == "High Achiever":
        defaults.update({
            "studytime": 4, "failures": 0, "higher": "yes", "Medu": 4, "Fedu": 4,
            "schoolsup": "no", "famsup": "yes", "internet": "yes", "absences": 1,
            "Dalc": 1, "Walc": 1, "freetime": 2, "goout": 2, "health": 5
        })
    elif preset == "At-Risk":
        defaults.update({
            "studytime": 1, "failures": 2, "higher": "no", "Medu": 1, "Fedu": 1,
            "schoolsup": "yes", "famsup": "no", "absences": 18,
            "Dalc": 3, "Walc": 4, "freetime": 4, "goout": 4, "health": 2
        })
        
    st.info(f"Active Profile Mode: **{preset}**")

    # The Form: 6 Organized Feature Groups
    with st.form("student_prediction_form"):
        # GROUP 1: Personal Demographics
        st.markdown("### 1. 👤 Personal Information")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            school = st.selectbox(
                "School Attended",
                options=["GP", "MS"],
                index=["GP", "MS"].index(defaults["school"]),
                help="GP = Gabriel Pereira, MS = Mousinho da Silveira"
            )
            sex = st.selectbox(
                "Gender",
                options=["F", "M"],
                index=["F", "M"].index(defaults["sex"]),
                format_func=lambda x: "Female" if x == "F" else "Male"
            )
        with p_col2:
            age = st.slider("Age (Years)", min_value=15, max_value=22, value=int(defaults["age"]))
            address = st.selectbox(
                "Home Address Type",
                options=["U", "R"],
                index=["U", "R"].index(defaults["address"]),
                format_func=lambda x: "Urban" if x == "U" else "Rural"
            )
        with p_col3:
            famsize = st.selectbox(
                "Family Size",
                options=["GT3", "LE3"],
                index=["GT3", "LE3"].index(defaults["famsize"]),
                format_func=lambda x: "More than 3 members (GT3)" if x == "GT3" else "Up to 3 members (LE3)"
            )
            pstatus = st.selectbox(
                "Parents' Cohabitation Status",
                options=["T", "A"],
                index=["T", "A"].index(defaults["Pstatus"]),
                format_func=lambda x: "Living Together (T)" if x == "T" else "Apart / Separated (A)"
            )
            
        st.markdown("---")
        
        # GROUP 2: Family Background
        st.markdown("### 2. 🏡 Family Background & Support")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            medu = st.selectbox(
                "Mother's Education Level",
                options=[0, 1, 2, 3, 4],
                index=defaults["Medu"],
                format_func=lambda x: {
                    0: "0 - None",
                    1: "1 - Primary (4th Grade)",
                    2: "2 - 5th to 9th Grade",
                    3: "3 - Secondary Education",
                    4: "4 - Higher Education"
                }[x]
            )
            fedu = st.selectbox(
                "Father's Education Level",
                options=[0, 1, 2, 3, 4],
                index=defaults["Fedu"],
                format_func=lambda x: {
                    0: "0 - None",
                    1: "1 - Primary (4th Grade)",
                    2: "2 - 5th to 9th Grade",
                    3: "3 - Secondary Education",
                    4: "4 - Higher Education"
                }[x]
            )
        with f_col2:
            mjob = st.selectbox(
                "Mother's Occupation",
                options=["teacher", "health", "services", "at_home", "other"],
                index=["teacher", "health", "services", "at_home", "other"].index(defaults["Mjob"]),
                format_func=lambda x: x.replace("_", " ").title()
            )
            fjob = st.selectbox(
                "Father's Occupation",
                options=["teacher", "health", "services", "at_home", "other"],
                index=["teacher", "health", "services", "at_home", "other"].index(defaults["Fjob"]),
                format_func=lambda x: x.replace("_", " ").title()
            )
        with f_col3:
            guardian = st.selectbox(
                "Legal Guardian",
                options=["mother", "father", "other"],
                index=["mother", "father", "other"].index(defaults["guardian"]),
                format_func=lambda x: x.title()
            )
            reason = st.selectbox(
                "Reason for Choosing School",
                options=["course", "home", "reputation", "other"],
                index=["course", "home", "reputation", "other"].index(defaults["reason"]),
                format_func=lambda x: {
                    "course": "Course Preference",
                    "home": "Close to Home",
                    "reputation": "School Reputation",
                    "other": "Other Reason"
                }[x]
            )
            
        st.markdown("---")
        
        # GROUP 3: Education & Study Habits
        st.markdown("### 3. 📖 Education & Study Habits")
        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                index=defaults["studytime"] - 1,
                format_func=lambda x: {
                    1: "1 - Less than 2 hours",
                    2: "2 - 2 to 5 hours",
                    3: "3 - 5 to 10 hours",
                    4: "4 - More than 10 hours"
                }[x]
            )
            failures = st.selectbox(
                "Past Class Failures",
                options=[0, 1, 2, 3],
                index=defaults["failures"],
                format_func=lambda x: f"{x} past failure(s)" if x < 3 else "3 or more past failures"
            )
        with e_col2:
            traveltime = st.selectbox(
                "Home to School Travel Time",
                options=[1, 2, 3, 4],
                index=defaults["traveltime"] - 1,
                format_func=lambda x: {
                    1: "1 - Less than 15 minutes",
                    2: "2 - 15 to 30 minutes",
                    3: "3 - 30 min to 1 hour",
                    4: "4 - More than 1 hour"
                }[x]
            )
            higher = st.selectbox(
                "Aspires to Pursue Higher Education",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["higher"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )
        with e_col3:
            famrel = st.slider(
                "Family Relationship Quality",
                min_value=1,
                max_value=5,
                value=4,
                help="1 = Very Bad, 5 = Excellent"
            )
            
        st.markdown("---")
        
        # GROUP 4: School & Educational Support
        st.markdown("### 4. 🏫 School Support & Environment")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            schoolsup = st.selectbox(
                "Extra Educational School Support",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["schoolsup"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )
            famsup = st.selectbox(
                "Family Educational Support",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["famsup"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )
        with s_col2:
            paid = st.selectbox(
                "Extra Paid Math Tutoring Classes",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["paid"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )
            activities = st.selectbox(
                "Extra-Curricular Activities",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["activities"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )
        with s_col3:
            nursery = st.selectbox(
                "Attended Nursery School",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["nursery"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )
            internet = st.selectbox(
                "Internet Access at Home",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["internet"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )

        st.markdown("---")
        
        # GROUP 5: Lifestyle & Health
        st.markdown("### 5. 🏃 Lifestyle, Social & Health")
        l_col1, l_col2, l_col3 = st.columns(3)
        with l_col1:
            freetime = st.slider("Free Time After School", min_value=1, max_value=5, value=int(defaults["freetime"]), help="1 = Very Low, 5 = Very High")
            goout = st.slider("Going Out with Friends", min_value=1, max_value=5, value=int(defaults["goout"]), help="1 = Very Low, 5 = Very High")
        with l_col2:
            dalc = st.slider("Workday Alcohol Consumption", min_value=1, max_value=5, value=int(defaults["Dalc"]), help="1 = Very Low, 5 = Very High")
            walc = st.slider("Weekend Alcohol Consumption", min_value=1, max_value=5, value=int(defaults["Walc"]), help="1 = Very Low, 5 = Very High")
        with l_col3:
            health = st.slider("Current Health Status", min_value=1, max_value=5, value=int(defaults["health"]), help="1 = Very Bad, 5 = Very Good")
            romantic = st.selectbox(
                "In a Romantic Relationship",
                options=["yes", "no"],
                index=["yes", "no"].index(defaults["romantic"]),
                format_func=lambda x: "Yes" if x == "yes" else "No"
            )
            
        st.markdown("---")
        
        # GROUP 6: Attendance
        st.markdown("### 6. 📅 School Attendance")
        a_col1, a_col2 = st.columns([1, 2])
        with a_col1:
            absences = st.number_input(
                "School Absences (Number of days)",
                min_value=0,
                max_value=93,
                value=int(defaults["absences"]),
                step=1,
                help="Historical dataset mean is approximately 5.7 days."
            )
        with a_col2:
            st.caption(
                """
                **Note on Attendance**: In the Random Forest feature analysis, absences rank as one of the 
                most influential variables. Moderately low absences correlate with solid engagement, 
                while extreme values (>15) frequently flag academic risk.
                """
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_submitted = st.form_submit_button(
            "🚀 Predict Final Mathematics Grade (G3)",
            use_container_width=True
        )

    # Prediction Execution & Results
    if predict_submitted:
        if rf_model is None:
            st.error(f"Cannot perform prediction because model file is unavailable: {model_error}")
        else:
            # Construct exact 30-feature single-row input DataFrame
            input_dict = {
                "school": school,
                "sex": sex,
                "age": int(age),
                "address": address,
                "famsize": famsize,
                "Pstatus": pstatus,
                "Medu": int(medu),
                "Fedu": int(fedu),
                "Mjob": mjob,
                "Fjob": fjob,
                "reason": reason,
                "guardian": guardian,
                "traveltime": int(traveltime),
                "studytime": int(studytime),
                "failures": int(failures),
                "schoolsup": schoolsup,
                "famsup": famsup,
                "paid": paid,
                "activities": activities,
                "nursery": nursery,
                "higher": higher,
                "internet": internet,
                "romantic": romantic,
                "famrel": int(famrel),
                "freetime": int(freetime),
                "goout": int(goout),
                "Dalc": int(dalc),
                "Walc": int(walc),
                "health": int(health),
                "absences": int(absences),
            }
            input_df = pd.DataFrame([input_dict])

            try:
                raw_pred = float(rf_model.predict(input_df)[0])
                # Portuguese grading scale is 0 to 20
                display_pred = max(0.0, min(20.0, raw_pred))
                
                # Determine Performance Tier
                if display_pred >= 16.0:
                    band_name = "Distinction / Excellent (16–20)"
                    band_class = "result-card-excellent"
                    band_desc = "Student demonstrates strong positive academic and lifestyle indicators."
                elif display_pred >= 14.0:
                    band_name = "Good / Above Average (14–15)"
                    band_class = "result-card-good"
                    band_desc = "Solid academic foundation with minor areas for potential optimization."
                elif display_pred >= 10.0:
                    band_name = "Satisfactory / Pass (10–13)"
                    band_class = "result-card-pass"
                    band_desc = "Passing grade expected; targeted support can elevate performance."
                else:
                    band_name = "At Risk / Intervention Advised (0–9)"
                    band_class = "result-card-risk"
                    band_desc = "Early intervention and mentoring recommended to prevent academic failure."

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">🎯 Prediction Outcome</div>', unsafe_allow_html=True)

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.markdown(
                        f"""
                        <div class="{band_class}">
                            <div style="font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                                Predicted Final Grade (G3)
                            </div>
                            <div style="font-size: 3.2rem; font-weight: 800; line-height: 1; margin-bottom: 0.5rem;">
                                {display_pred:.2f} <span style="font-size: 1.5rem; font-weight: 500;">/ 20</span>
                            </div>
                            <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem;">
                                {band_name}
                            </div>
                            <div style="font-size: 0.9rem; line-height: 1.4; opacity: 0.9;">
                                {band_desc}
                            </div>
                            <hr style="margin: 1rem 0; opacity: 0.3;">
                            <div style="font-size: 0.8rem;">
                                <strong>Raw Pipeline Value:</strong> {raw_pred:.3f} | 
                                <strong>Dataset Mean:</strong> 10.42 / 20
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with res_col2:
                    # Plotly Visual Gauge
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=display_pred,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Grade Indicator (0–20 Scale)", "font": {"size": 16, "color": "#0F172A"}},
                            delta={"reference": 10.42, "increasing": {"color": "#0D9488"}, "decreasing": {"color": "#E11D48"}},
                            gauge={
                                "axis": {"range": [0, 20], "tickwidth": 1, "tickcolor": "#94A3B8"},
                                "bar": {"color": "#0F172A", "thickness": 0.28},
                                "bgcolor": "white",
                                "borderwidth": 1,
                                "bordercolor": "#CBD5E1",
                                "steps": [
                                    {"range": [0, 9.99], "color": "#FEE2E2"},
                                    {"range": [10, 13.99], "color": "#DBEAFE"},
                                    {"range": [14, 15.99], "color": "#CCFBF1"},
                                    {"range": [16, 20], "color": "#D1FAE5"},
                                ],
                                "threshold": {
                                    "line": {"color": "#DC2626", "width": 3},
                                    "thickness": 0.75,
                                    "value": 10.0,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Factor Impact Analysis
                st.markdown("#### 🔍 Student Profile Key Drivers")
                f1, f2, f3 = st.columns(3)
                with f1:
                    study_strength = "Strong" if studytime >= 3 else ("Adequate" if studytime == 2 else "Low")
                    st.metric("Study Commitment", study_strength, f"{studytime} Level")
                with f2:
                    fail_status = "Clean Record" if failures == 0 else f"{failures} Failures"
                    st.metric("Past Academic Trajectory", fail_status, "Zero Failures" if failures == 0 else "Risk Factor")
                with f3:
                    abs_eval = "Good" if absences <= 4 else ("Moderate" if absences <= 10 else "High")
                    st.metric("Absence Impact", abs_eval, f"{absences} Days")

                st.markdown(
                    """
                    <div style="background: #F1F5F9; border-left: 4px solid #0D9488; padding: 0.9rem 1.2rem; border-radius: 4px; font-size: 0.85rem; color: #334155; margin-top: 1rem;">
                        <strong>Academic Evaluation Note:</strong> This grade prediction is produced by a Random Forest 
                        regression model trained on 395 historical student records from the UCI Machine Learning Repository. 
                        It exhibits a Mean Absolute Error of ±2.998 grade points on unseen test records and is meant solely 
                        for advisory and academic demonstration purposes.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as ex:
                st.error(f"Prediction processing error: {str(ex)}")

# -----------------------------------------------------------------------------
# 6. VIEW: 📊 ANALYTICS DASHBOARD
# -----------------------------------------------------------------------------
elif nav_selection == "📊 Analytics Dashboard":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <div class="hero-badge">Model Evaluation & Validation</div>
            <div class="hero-title" style="font-size: 1.8rem;">Machine Learning Analytics Dashboard</div>
            <div class="hero-subtitle">
                Inspect empirical evaluation metrics, algorithm comparisons, feature importance distributions, 
                and test set residual analysis verified across experimental training runs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # KPI Performance Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Mean Absolute Error</div>
                <div class="metric-value">{VERIFIED_METRICS['rf_mae']}</div>
                <div class="metric-caption">Average error on unseen test set</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Root Mean Squared Error</div>
                <div class="metric-value">{VERIFIED_METRICS['rf_rmse']}</div>
                <div class="metric-caption">Penalizes larger grade errors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test $R^2$ Score</div>
                <div class="metric-value">{VERIFIED_METRICS['rf_r2']}</div>
                <div class="metric-caption">Non-exam baseline variance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">5-Fold Cross-Val $R^2$</div>
                <div class="metric-value">{VERIFIED_METRICS['rf_cv_r2']}</div>
                <div class="metric-caption">Mean across 5 stratified folds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Comparison Section
    st.markdown('<div class="section-header">⚖️ Algorithmic Comparison</div>', unsafe_allow_html=True)
    
    comp_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest (Baseline)", "Random Forest (Tuned)"],
        "MAE (Lower is Better)": [VERIFIED_METRICS["lr_mae"], VERIFIED_METRICS["rf_mae"], VERIFIED_METRICS["tuned_rf_mae"]],
        "RMSE (Lower is Better)": [VERIFIED_METRICS["lr_rmse"], VERIFIED_METRICS["rf_rmse"], VERIFIED_METRICS["tuned_rf_rmse"]],
        "R² Score (Higher is Better)": [VERIFIED_METRICS["lr_r2"], VERIFIED_METRICS["rf_r2"], VERIFIED_METRICS["tuned_rf_r2"]],
    })
    
    comp_col1, comp_col2 = st.columns([1, 1.3])
    with comp_col1:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption(
            """
            **Evaluation Analysis**: The Random Forest regressor out-performs Linear Regression across 
            every benchmark metric, capturing non-linear feature interactions (such as the compounded effect 
            of high absences combined with past failures). Hyperparameter tuning via GridSearchCV yielded 
            a slightly more conservative model ($R^2 = 0.254$), making the original Random Forest configuration 
            the best generalization model on the test split.
            """
        )
    with comp_col2:
        # Grouped bar chart comparing metrics
        melted_comp = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_comp = px.bar(
            melted_comp,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_sequence=["#94A3B8", "#0D9488", "#0284C7"],
            title="Comparison of Model Evaluation Metrics",
        )
        fig_comp.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations: Actual vs Predicted and Feature Importance
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown('<div class="section-header">📈 Actual vs. Predicted Grades</div>', unsafe_allow_html=True)
        if df_test_results is not None:
            fig_scatter = px.scatter(
                df_test_results,
                x="Actual Grade",
                y="RF Predicted Grade",
                hover_data=["RF Absolute Error"],
                color="RF Absolute Error",
                color_continuous_scale="Tealgrn",
                labels={"Actual Grade": "Actual G3 Grade (0–20)", "RF Predicted Grade": "Predicted G3 Grade (0–20)"},
                title="Random Forest Test Set Predictions (79 Samples)",
            )
            # Add ideal parity reference line
            fig_scatter.add_shape(
                type="line",
                x0=0, y0=0, x1=20, y1=20,
                line=dict(color="#DC2626", dash="dash", width=2),
            )
            fig_scatter.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Red dashed line represents perfect prediction ($y = x$).")
        else:
            st.info("Generating test scatter plot...")

    with v_col2:
        st.markdown('<div class="section-header">🏆 Top 10 Influential Features</div>', unsafe_allow_html=True)
        if df_importance is not None:
            top_10_imp = df_importance.head(10).sort_values("Importance", ascending=True)
            # Clean feature names for clean presentation
            top_10_imp["Clean_Feature"] = top_10_imp["Feature"].apply(
                lambda x: x.replace("num__", "").replace("cat__", "").replace("_", " ").title()
            )
            fig_imp = px.bar(
                top_10_imp,
                x="Importance",
                y="Clean_Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                title="Gini Impurity Feature Importance",
            )
            fig_imp.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Absences, failures, and parent education emerge as leading predictive factors.")
        else:
            st.info("Feature importance data is loading...")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Residual Error Analysis
    err_col1, err_col2 = st.columns(2)
    
    with err_col1:
        st.markdown('<div class="section-header">📉 Prediction Error Distribution</div>', unsafe_allow_html=True)
        if df_test_results is not None:
            residuals = df_test_results["Actual Grade"] - df_test_results["RF Predicted Grade"]
            fig_hist = px.histogram(
                x=residuals,
                nbins=15,
                labels={"x": "Prediction Error (Actual - Predicted)"},
                title="Distribution of Test Residuals",
                color_discrete_sequence=["#0D9488"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#0F172A", annotation_text="Zero Error")
            fig_hist.add_vline(x=-2.998, line_dash="dot", line_color="#DC2626")
            fig_hist.add_vline(x=2.998, line_dash="dot", line_color="#DC2626", annotation_text="±MAE Boundary")
            fig_hist.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True)
            within_two = (np.abs(residuals) <= 2).mean() * 100
            st.caption(f"**Accuracy Concentration**: {within_two:.1f}% of all test predictions fall within ±2 grade points.")
    
    with err_col2:
        st.markdown('<div class="section-header">📊 Error Across Actual Grade Ranges</div>', unsafe_allow_html=True)
        if df_test_results is not None:
            # Recreate verified error analysis by grade range
            df_bins = df_test_results.copy()
            df_bins["Grade Range"] = pd.cut(
                df_bins["Actual Grade"],
                bins=[-1, 5, 10, 15, 20],
                labels=["0–5", "6–10", "11–15", "16–20"]
            )
            tier_analysis = df_bins.groupby("Grade Range", observed=False)["RF Absolute Error"].mean().reset_index()
            tier_analysis.columns = ["Grade Tier", "Mean Absolute Error"]
            
            fig_tier = px.bar(
                tier_analysis,
                x="Grade Tier",
                y="Mean Absolute Error",
                color="Mean Absolute Error",
                color_continuous_scale="Reds",
                title="Mean Absolute Error by Actual Grade Tier",
            )
            fig_tier.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tier, use_container_width=True)
            st.caption("Prediction error is lowest in the 11–15 range; extreme grades exhibit regression-to-mean effect.")

    # Dataset Distribution Explorations
    if df_raw is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📚 Dataset Exploratory Distributions</div>', unsafe_allow_html=True)
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            fig_g3_dist = px.histogram(
                df_raw,
                x="G3",
                nbins=21,
                title="Distribution of Final Grades (G3) in Entire Dataset",
                labels={"G3": "Final Grade (0–20)"},
                color_discrete_sequence=["#0284C7"]
            )
            fig_g3_dist.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g3_dist, use_container_width=True)
            st.caption("Notice the distinct cluster at 0 (representing course dropouts or exam absentees).")
        with d_col2:
            fig_study_box = px.box(
                df_raw,
                x="studytime",
                y="G3",
                title="Study Time Category vs. Final Mathematics Grade",
                labels={"studytime": "Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", "G3": "Final Grade"},
                color="studytime",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_study_box.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_study_box, use_container_width=True)
            st.caption("Higher study commitment shows an upward shift in median grade achievement.")

# -----------------------------------------------------------------------------
# 7. VIEW: 📖 ABOUT PROJECT
# -----------------------------------------------------------------------------
elif nav_selection == "📖 About Project":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <div class="hero-badge">Academic Documentation</div>
            <div class="hero-title" style="font-size: 1.8rem;">Project Methodology & Technical Architecture</div>
            <div class="hero-subtitle">
                Comprehensive documentation covering the problem statement, dataset architecture, 
                feature engineering rationale, validation methodologies, limitations, and future extensions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### 1. Problem Statement & Motivation")
    st.write(
        """
        Secondary education student achievement in mathematics is heavily influenced by non-academic variables, 
        such as family socio-economic status, personal study discipline, peer interactions, and lifestyle choices. 
        Traditional academic intervention systems often react only after mid-term examination failures ($G1$ or $G2$). 
        
        The objective of this project is to build an early predictive system that forecasts the final grade ($G3$) 
        using **exclusively pre-examination and lifestyle indicators**, enabling instructors and academic mentors 
        to identify at-risk students well before the first examination period.
        """
    )
    
    st.markdown("### 2. Dataset Information")
    st.write(
        """
        The data was sourced from the **UCI Machine Learning Repository: Student Performance Data Set** 
        (Cortez and Silva, 2008). It gathers data from two Portuguese public secondary schools (Gabriel Pereira 
        and Mousinho da Silveira) during the 2005–2006 academic year:
        
        - **Source URL**: [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
        - **Subject Focus**: Mathematics course (`student-mat.csv`)
        - **Total Records**: 395 student instances
        - **Total Features**: 33 original attributes (30 predictor features + $G1, G2, G3$)
        - **Target Variable**: $G3$ — Final year grade on a scale from 0 to 20.
        """
    )
    
    st.markdown("### 3. Feature Selection: Why Exclude G1 and G2?")
    st.write(
        """
        While historical models often achieve higher $R^2$ values (>0.80) when including $G1$ (first period) and $G2$ (second period) grades, 
        doing so causes **target leakage for early intervention**:
        
        1. **Practical Timing**: If educators already have $G1$ and $G2$, the academic semester is already 70% complete.
        2. **True Predictive Challenge**: Predicting final achievement using strictly socio-demographic, attendance, and lifestyle 
           indicators poses an authentic machine learning challenge and produces a model viable from the first day of class.
        """
    )
    
    st.markdown("### 4. Machine Learning Pipeline Architecture")
    st.write(
        """
        The system implements a production-grade Scikit-learn Pipeline architecture to eliminate data leakage:
        
        - **Train/Test Splitting**: 80% training (316 records) and 20% testing (79 records) with fixed `random_state=42`.
        - **ColumnTransformer Preprocessing**:
          - **17 Categorical Features**: Encoded using `OneHotEncoder(handle_unknown='ignore')`.
          - **13 Numerical Features**: Maintained using `passthrough`.
        - **Regression Algorithm**: `RandomForestRegressor(n_estimators=100, random_state=42)`.
        - **Serialization**: Saved as portable Scikit-learn pipeline objects using `joblib`.
        """
    )
    
    st.markdown("### 5. Verified Experimental Results")
    res_table = pd.DataFrame({
        "Configuration": ["Linear Regression Pipeline", "Random Forest Regressor (Selected)", "Tuned Random Forest (GridSearchCV)"],
        "MAE": [3.395, 2.998, 3.108],
        "RMSE": [4.196, 3.795, 3.910],
        "R² Score": [0.142, 0.298, 0.254],
        "CV Mean R²": ["N/A", "0.273 (5-fold)", "0.248 (3-fold)"]
    })
    st.table(res_table)
    
    st.markdown("### 6. Limitations & Honest Academic Discussion")
    st.write(
        """
        - **Modest $R^2$ Value (0.298)**: Human cognitive performance and mathematical ability are shaped by 
          unmeasured factors like individual psychology, cognitive aptitude, teacher quality, and syllabus difficulty. 
          Without prior exam scores, an $R^2$ near 0.30 represents a realistic, uninflated behavioral benchmark.
        - **Bimodal Grade 0 Distribution**: In the Portuguese grading structure, students who drop out or miss 
          the final exam receive an automatic 0. Standard regression models predict the expected value (~9.5), 
          which creates higher absolute residuals for true zeros.
        - **Sample Size**: With 395 total records, statistical generalizability to non-European school systems 
          requires external validation.
        """
    )
    
    st.markdown("### 7. Future Scope & Roadmap")
    st.write(
        """
        1. **Two-Stage Hurdle Model**: Combine a binary classifier (predicting exam non-attendance/dropout: 0 vs. >0) 
           with a regressor (predicting grades 1–20 for active students).
        2. **Longitudinal Cross-Subject Modeling**: Integrate the Portuguese language dataset (`student-por.csv`) 
           to perform multi-task transfer learning.
        3. **Explainable AI (SHAP / LIME)**: Add local feature attribution for personalized student diagnostic reports.
        """
    )

# -----------------------------------------------------------------------------
# 8. APPLICATION FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <p style="font-weight: 600; color: #0F172A; margin-bottom: 0.3rem;">
            Student Performance Prediction System
        </p>
        <p style="margin-bottom: 0.6rem;">
            Developed as an academic machine learning mini-project for IT Portfolio & Internship Demonstration.
        </p>
        <div>
            <span class="footer-badge">Python 3.13</span>
            <span class="footer-badge">Scikit-Learn</span>
            <span class="footer-badge">Streamlit</span>
            <span class="footer-badge">Pandas</span>
            <span class="footer-badge">Plotly</span>
            <span class="footer-badge">Random Forest</span>
        </div>
        <p style="margin-top: 0.8rem; font-size: 0.82rem;">
            GitHub Repository: <a href="https://github.com/Tejaswini-2006/student-performance-mini-project" target="_blank" style="color: #0D9488; text-decoration: none; font-weight: 600;">Tejaswini-2006/student-performance-mini-project</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
