"""
Student Performance Dashboard
=============================
A modern, unified, single-page student performance analytics and ML prediction dashboard.
Predicts secondary school mathematics final grade (G3, 0-20 scale) using 30 socio-demographic features.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & MINIMALIST SAAS STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DASHBOARD_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Clean subtle background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Hide default streamlit decorations */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Top Dashboard Header */
    .dash-topbar {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.05);
    }

    .dash-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .dash-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        margin: 0.2rem 0 0 0;
    }

    .status-badge {
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 6px -2px rgba(15, 23, 42, 0.04);
        transition: all 0.15s ease;
    }

    .kpi-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 6px 14px -4px rgba(15, 23, 42, 0.08);
    }

    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: #64748B;
        margin-bottom: 0.3rem;
    }

    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.78rem;
        color: #0D9488;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Main Content Cards */
    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.25rem;
    }

    .panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result Pill Cards */
    .result-box {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-excellent {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #6EE7B7;
        color: #065F46;
    }

    .result-good {
        background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
        border: 1px solid #5EEAD4;
        color: #0F766E;
    }

    .result-pass {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #93C5FD;
        color: #1E40AF;
    }

    .result-risk {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #FCA5A5;
        color: #991B1B;
    }

    .grade-display {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.4rem 0;
    }

    /* Primary Button Styling */
    div.stButton > button:first-child {
        background: #0F172A;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.25rem;
        transition: all 0.2s ease;
    }

    div.stButton > button:first-child:hover {
        background: #1E293B;
        color: #38BDF8;
        transform: translateY(-1px);
    }

    /* Minimal Footer */
    .mini-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.8rem;
        padding: 1.5rem 0 1rem 0;
        border-top: 1px solid #E2E8F0;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL & DATA CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model = joblib.load("student_performance_random_forest.pkl") if os.path.exists("student_performance_random_forest.pkl") else None
    dataset = pd.read_csv("student-mat.csv", sep=";") if os.path.exists("student-mat.csv") else None
    importance = pd.read_csv("student_feature_importance.csv") if os.path.exists("student_feature_importance.csv") else None
    results = pd.read_csv("student_prediction_results.csv") if os.path.exists("student_prediction_results.csv") else None
    return model, dataset, importance, results

rf_model, df_raw, df_importance, df_results = load_assets()

# -----------------------------------------------------------------------------
# 3. TOP DASHBOARD HEADER & KPI BAR
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="dash-topbar">
        <div>
            <h1 class="dash-title">🎓 Student Performance Dashboard</h1>
            <p class="dash-subtitle">Predictive Grade Intelligence & Academic Analytics · Scale 0–20</p>
        </div>
        <div class="status-badge">
            <span style="font-size: 0.9rem;">●</span> Random Forest Pipeline Active
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 Key Metric Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Test Set MAE</div>
            <div class="kpi-value">2.998</div>
            <div class="kpi-sub">Mean Absolute Error</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Variance R² Score</div>
            <div class="kpi-value">0.298</div>
            <div class="kpi-sub">Cross-Val R²: 0.273</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Dataset Records</div>
            <div class="kpi-value">{len(df_raw) if df_raw is not None else 395}</div>
            <div class="kpi-sub">UCI Mathematics Benchmark</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Input Predictors</div>
            <div class="kpi-value">30</div>
            <div class="kpi-sub">Excludes Midterms G1 & G2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD: 2-COLUMN UNIFIED VIEW
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.08, 0.92], gap="large")

# -----------------------------------------------------------------------------
# COLUMN 1: LIVE PREDICTION ENGINE
# -----------------------------------------------------------------------------
with left_col:
    st.markdown('<div class="panel-title">⚡ Student Grade Predictor</div>', unsafe_allow_html=True)
    
    # Preset Quick Buttons
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    if "dash_preset" not in st.session_state:
        st.session_state.dash_preset = "Average"
        
    with preset_col1:
        if st.button("🌟 Top Achiever", use_container_width=True):
            st.session_state.dash_preset = "Top"
            st.rerun()
    with preset_col2:
        if st.button("📘 Average", use_container_width=True):
            st.session_state.dash_preset = "Average"
            st.rerun()
    with preset_col3:
        if st.button("⚠️ At Risk", use_container_width=True):
            st.session_state.dash_preset = "Risk"
            st.rerun()
    with preset_col4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.dash_preset = "Average"
            st.rerun()

    cur_p = st.session_state.dash_preset
    
    # Default profiles
    form_defaults = {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 3, "Fedu": 3, "Mjob": "services", "Fjob": "other", "reason": "course", "guardian": "mother",
        "traveltime": 1, "studytime": 2, "failures": 0, "higher": "yes",
        "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "yes", "nursery": "yes", "internet": "yes",
        "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4, "romantic": "no",
        "absences": 4
    }
    
    if cur_p == "Top":
        form_defaults.update({
            "studytime": 4, "failures": 0, "higher": "yes", "Medu": 4, "Fedu": 4,
            "schoolsup": "no", "famsup": "yes", "internet": "yes", "absences": 1,
            "Dalc": 1, "Walc": 1, "freetime": 2, "goout": 2, "health": 5
        })
    elif cur_p == "Risk":
        form_defaults.update({
            "studytime": 1, "failures": 2, "higher": "no", "Medu": 1, "Fedu": 1,
            "schoolsup": "yes", "famsup": "no", "absences": 18,
            "Dalc": 3, "Walc": 4, "freetime": 4, "goout": 4, "health": 2
        })

    # The Prediction Form
    with st.form("quick_prediction_form"):
        # Section A: Key Academic Habits
        st.markdown("**1. Academics & Attendance**")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4],
                index=form_defaults["studytime"] - 1,
                format_func=lambda x: {1: "< 2 hrs/wk", 2: "2 - 5 hrs/wk", 3: "5 - 10 hrs/wk", 4: "> 10 hrs/wk"}[x]
            )
        with ac2:
            failures = st.selectbox(
                "Past Class Failures",
                [0, 1, 2, 3],
                index=form_defaults["failures"],
                format_func=lambda x: f"{x} Failures" if x < 3 else "3+ Failures"
            )
        with ac3:
            absences = st.number_input(
                "Absence Days",
                min_value=0,
                max_value=93,
                value=int(form_defaults["absences"]),
                step=1
            )

        ac4, ac5, ac6 = st.columns(3)
        with ac4:
            higher = st.selectbox("Higher Education Plans", ["yes", "no"], index=["yes", "no"].index(form_defaults["higher"]))
        with ac5:
            traveltime = st.selectbox(
                "Travel Time",
                [1, 2, 3, 4],
                index=form_defaults["traveltime"] - 1,
                format_func=lambda x: {1: "< 15 min", 2: "15-30 min", 3: "30-60 min", 4: "> 1 hr"}[x]
            )
        with ac6:
            paid = st.selectbox("Extra Paid Classes", ["no", "yes"], index=["no", "yes"].index(form_defaults["paid"]))

        st.markdown("**2. Demographics & Family**")
        df1, df2, df3, df4 = st.columns(4)
        with df1:
            age = st.slider("Age", 15, 22, int(form_defaults["age"]))
        with df2:
            sex = st.selectbox("Gender", ["F", "M"], index=["F", "M"].index(form_defaults["sex"]))
        with df3:
            school = st.selectbox("School", ["GP", "MS"], index=["GP", "MS"].index(form_defaults["school"]))
        with df4:
            address = st.selectbox("Address", ["U", "R"], index=["U", "R"].index(form_defaults["address"]), format_func=lambda x: "Urban" if x == "U" else "Rural")

        df5, df6, df7, df8 = st.columns(4)
        with df5:
            medu = st.selectbox("Mother Edu", [0, 1, 2, 3, 4], index=form_defaults["Medu"], format_func=lambda x: f"Level {x}")
        with df6:
            fedu = st.selectbox("Father Edu", [0, 1, 2, 3, 4], index=form_defaults["Fedu"], format_func=lambda x: f"Level {x}")
        with df7:
            mjob = st.selectbox("Mother Job", ["services", "health", "teacher", "at_home", "other"], index=0)
        with df8:
            fjob = st.selectbox("Father Job", ["other", "services", "health", "teacher", "at_home"], index=0)

        # Section C: Lifestyle & Well-being
        with st.expander("⚙️ Additional Lifestyle & Social Features (Click to adjust)"):
            ls1, ls2, ls3 = st.columns(3)
            with ls1:
                freetime = st.slider("Free Time (1-5)", 1, 5, int(form_defaults["freetime"]))
                goout = st.slider("Going Out (1-5)", 1, 5, int(form_defaults["goout"]))
            with ls2:
                dalc = st.slider("Workday Alcohol (1-5)", 1, 5, int(form_defaults["Dalc"]))
                walc = st.slider("Weekend Alcohol (1-5)", 1, 5, int(form_defaults["Walc"]))
            with ls3:
                health = st.slider("Health Status (1-5)", 1, 5, int(form_defaults["health"]))
                famrel = st.slider("Family Relationship (1-5)", 1, 5, 4)

            ls4, ls5, ls6, ls7 = st.columns(4)
            with ls4:
                schoolsup = st.selectbox("School Support", ["no", "yes"], index=0)
            with ls5:
                famsup = st.selectbox("Family Support", ["yes", "no"], index=0)
            with ls6:
                internet = st.selectbox("Internet Access", ["yes", "no"], index=0)
            with ls7:
                romantic = st.selectbox("Romantic Relation", ["no", "yes"], index=0)

            ls8, ls9, ls10, ls11 = st.columns(4)
            with ls8:
                famsize = st.selectbox("Family Size", ["GT3", "LE3"], index=0)
            with ls9:
                pstatus = st.selectbox("Parent Status", ["T", "A"], index=0)
            with ls10:
                guardian = st.selectbox("Guardian", ["mother", "father", "other"], index=0)
            with ls11:
                reason = st.selectbox("School Reason", ["course", "home", "reputation", "other"], index=0)
                
            activities = "yes"
            nursery = "yes"

        predict_btn = st.form_submit_button("🔮 Predict Student Grade (G3)", use_container_width=True)

    # Prediction Outcome Render
    if predict_btn and rf_model is not None:
        row_dict = {
            "school": school, "sex": sex, "age": int(age), "address": address,
            "famsize": famsize, "Pstatus": pstatus, "Medu": int(medu), "Fedu": int(fedu),
            "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
            "traveltime": int(traveltime), "studytime": int(studytime), "failures": int(failures),
            "schoolsup": schoolsup, "famsup": famsup, "paid": paid, "activities": activities,
            "nursery": nursery, "higher": higher, "internet": internet, "romantic": romantic,
            "famrel": int(famrel), "freetime": int(freetime), "goout": int(goout),
            "Dalc": int(dalc), "Walc": int(walc), "health": int(health), "absences": int(absences)
        }
        
        sample_df = pd.DataFrame([row_dict])
        pred_val = float(rf_model.predict(sample_df)[0])
        clamped_score = max(0.0, min(20.0, pred_val))
        
        if clamped_score >= 16.0:
            tier_class = "result-excellent"
            tier_badge = "🌟 Distinction / Excellent"
            tier_msg = "Outstanding academic trajectory with high probability of top exam performance."
        elif clamped_score >= 14.0:
            tier_class = "result-good"
            tier_badge = "✨ Good Standing"
            tier_msg = "Strong student profile with consistent passing performance expected."
        elif clamped_score >= 10.0:
            tier_class = "result-pass"
            tier_badge = "📘 Satisfactory / Pass"
            tier_msg = "Predicted to meet passing standards. Structured study routine will provide stability."
        else:
            tier_class = "result-risk"
            tier_badge = "⚠️ Academic Intervention Needed"
            tier_msg = "Below passing threshold (10/20). Early academic tutoring and attendance monitoring recommended."

        st.markdown(
            f"""
            <div class="result-box {tier_class}">
                <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                    Predicted Mathematics Grade (G3)
                </div>
                <div class="grade-display">
                    {clamped_score:.2f} <span style="font-size: 1.5rem; font-weight: 500;">/ 20</span>
                </div>
                <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.3rem;">
                    {tier_badge}
                </div>
                <div style="font-size: 0.85rem; opacity: 0.9;">
                    {tier_msg}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# COLUMN 2: ANALYTICS & VISUAL INSIGHTS
# -----------------------------------------------------------------------------
with right_col:
    st.markdown('<div class="panel-title">📊 Live Academic Analytics</div>', unsafe_allow_html=True)

    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
        "🏆 Feature Drivers",
        "📈 Actual vs Predicted",
        "📚 Grade Distributions"
    ])

    with analytics_tab1:
        if df_importance is not None:
            top_10 = df_importance.head(10).sort_values("Importance", ascending=True)
            top_10["Clean_Name"] = top_10["Feature"].apply(
                lambda x: x.replace("num__", "").replace("cat__", "").replace("_", " ").title()
            )
            fig_imp = px.bar(
                top_10,
                x="Importance",
                y="Clean_Name",
                orientation="h",
                color="Importance",
                color_continuous_scale="Teal",
                title="Top 10 Most Influential Student Performance Predictors",
            )
            fig_imp.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=35, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis_title="Relative Feature Importance",
                yaxis_title=""
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("🔍 **Insight**: School absences, past failures, and parental education dominate grade prediction.")

    with analytics_tab2:
        if df_results is not None:
            fig_scatter = px.scatter(
                df_results,
                x="Actual Grade",
                y="RF Predicted Grade",
                color="RF Absolute Error",
                color_continuous_scale="Viridis",
                hover_data=["RF Absolute Error"],
                title="Model Test Parity (79 Held-Out Students)",
            )
            fig_scatter.add_shape(
                type="line",
                x0=0, y0=0, x1=20, y1=20,
                line=dict(color="#DC2626", dash="dash", width=2)
            )
            fig_scatter.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=35, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Actual Final Grade (0-20)",
                yaxis_title="Predicted Final Grade"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("🎯 **Evaluation**: Red dashed line denotes ideal parity ($y = x$). Over 43% of predictions fall within ±2 grade points.")

    with analytics_tab3:
        if df_raw is not None:
            fig_box = px.box(
                df_raw,
                x="studytime",
                y="G3",
                color="studytime",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Weekly Study Time vs. Final Grade Distribution",
                labels={"studytime": "Study Time Tier (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", "G3": "Final Grade"}
            )
            fig_box.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=35, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("📈 **Trend**: Students studying >5 hours weekly show consistently higher median scores.")

# -----------------------------------------------------------------------------
# 5. MINIMAL CLEAN FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="mini-footer">
        <strong>Student Performance Dashboard</strong> • Machine Learning Mini-Project • 
        Dataset: UCI Math Dataset • GitHub: <a href="https://github.com/Tejaswini-2006/student-performance-mini-project" target="_blank" style="color: #0D9488; text-decoration: none; font-weight: 600;">Tejaswini-2006/student-performance-mini-project</a>
    </div>
    """,
    unsafe_allow_html=True,
)
