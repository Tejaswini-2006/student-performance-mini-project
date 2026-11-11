"""
Student Performance Dashboard
=============================
A simple, beginner-friendly Streamlit web app that predicts student mathematics
grades (0-20 scale) using a trained Random Forest model and displays interactive analytics.
"""

import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 2. LOAD TRAINED MODEL AND DATA (Cached for speed)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load the pre-trained Scikit-learn Random Forest Pipeline."""
    if os.path.exists("student_performance_random_forest.pkl"):
        return joblib.load("student_performance_random_forest.pkl")
    return None

@st.cache_data
def get_data():
    """Load dataset, test results, and feature importances."""
    df = pd.read_csv("student-mat.csv", sep=";") if os.path.exists("student-mat.csv") else None
    importance = pd.read_csv("student_feature_importance.csv") if os.path.exists("student_feature_importance.csv") else None
    results = pd.read_csv("student_prediction_results.csv") if os.path.exists("student_prediction_results.csv") else None
    return df, importance, results

model = get_model()
df, df_importance, df_results = get_data()

# -----------------------------------------------------------------------------
# 3. HEADER & QUICK METRICS
# -----------------------------------------------------------------------------
st.title("🎓 Student Performance Dashboard")
st.caption("Predict final mathematics grade (G3, 0–20 scale) using 30 student attributes · Random Forest Model")

# 4 Key Metrics Bar
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Model MAE", value="2.998", delta="-0.397 vs Linear Reg", delta_color="inverse")
m2.metric(label="Model R² Score", value="0.298", delta="5-Fold CV: 0.273")
m3.metric(label="Total Students", value=len(df) if df is not None else "395")
m4.metric(label="Input Features", value="30 Features", delta="Excludes G1 & G2", delta_color="off")

st.divider()

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD: PREDICTION (LEFT) & ANALYTICS (RIGHT)
# -----------------------------------------------------------------------------
col_pred, col_charts = st.columns([1, 1], gap="large")

# ----------------------- LEFT COLUMN: PREDICTOR -----------------------
with col_pred:
    st.subheader("⚡ Predict Student Grade")

    # Quick Preset Buttons for easy demo in college viva
    st.write("**Quick Demo Profiles:**")
    b1, b2, b3, b4 = st.columns(4)
    
    if "preset" not in st.session_state:
        st.session_state.preset = "Average"

    if b1.button("🌟 Top Student", use_container_width=True):
        st.session_state.preset = "Top"
        st.rerun()
    if b2.button("📘 Average", use_container_width=True):
        st.session_state.preset = "Average"
        st.rerun()
    if b3.button("⚠️ At Risk", use_container_width=True):
        st.session_state.preset = "Risk"
        st.rerun()
    if b4.button("🔄 Reset", use_container_width=True):
        st.session_state.preset = "Average"
        st.rerun()

    profile = st.session_state.preset

    # Default profile values
    vals = {
        "studytime": 2, "failures": 0, "absences": 4, "Medu": 3, "Fedu": 3,
        "higher": "yes", "age": 17, "sex": "F", "school": "GP", "address": "U",
        "traveltime": 1, "paid": "no", "freetime": 3, "goout": 3, "Dalc": 1,
        "Walc": 2, "health": 4
    }

    if profile == "Top":
        vals.update({
            "studytime": 4, "failures": 0, "absences": 1, "Medu": 4, "Fedu": 4,
            "higher": "yes", "freetime": 2, "goout": 2, "Dalc": 1, "Walc": 1, "health": 5
        })
    elif profile == "Risk":
        vals.update({
            "studytime": 1, "failures": 2, "absences": 18, "Medu": 1, "Fedu": 1,
            "higher": "no", "freetime": 4, "goout": 4, "Dalc": 3, "Walc": 4, "health": 2
        })

    # The Input Form
    with st.form("student_form"):
        st.write("##### 1. Academic & Study Habits")
        c1, c2, c3 = st.columns(3)
        studytime = c1.selectbox("Weekly Study Time", [1, 2, 3, 4], index=vals["studytime"] - 1,
                                 format_func=lambda x: {1: "< 2 hrs", 2: "2 - 5 hrs", 3: "5 - 10 hrs", 4: "> 10 hrs"}[x])
        failures = c2.selectbox("Past Failures", [0, 1, 2, 3], index=vals["failures"])
        absences = c3.number_input("Absence Days", min_value=0, max_value=93, value=vals["absences"], step=1)

        c4, c5, c6 = st.columns(3)
        higher = c4.selectbox("Wants Higher Education", ["yes", "no"], index=0 if vals["higher"] == "yes" else 1)
        traveltime = c5.selectbox("Travel Time", [1, 2, 3, 4], index=vals["traveltime"] - 1,
                                  format_func=lambda x: {1: "< 15 min", 2: "15-30 min", 3: "30-60 min", 4: "> 1 hr"}[x])
        paid = c6.selectbox("Paid Extra Classes", ["no", "yes"], index=0 if vals["paid"] == "no" else 1)

        st.write("##### 2. Student & Family Background")
        d1, d2, d3, d4 = st.columns(4)
        age = d1.slider("Age", 15, 22, vals["age"])
        sex = d2.selectbox("Gender", ["F", "M"], index=0 if vals["sex"] == "F" else 1)
        school = d3.selectbox("School", ["GP", "MS"], index=0 if vals["school"] == "GP" else 1)
        address = d4.selectbox("Address", ["U", "R"], index=0 if vals["address"] == "U" else 1,
                               format_func=lambda x: "Urban" if x == "U" else "Rural")

        d5, d6, d7, d8 = st.columns(4)
        medu = d5.selectbox("Mother Edu", [0, 1, 2, 3, 4], index=vals["Medu"], format_func=lambda x: f"Level {x}")
        fedu = d6.selectbox("Father Edu", [0, 1, 2, 3, 4], index=vals["Fedu"], format_func=lambda x: f"Level {x}")
        mjob = d7.selectbox("Mother Job", ["services", "health", "teacher", "at_home", "other"])
        fjob = d8.selectbox("Father Job", ["other", "services", "health", "teacher", "at_home"])

        # Extra Lifestyle options in an expander to keep it clean
        with st.expander("More Lifestyle Options (Optional)"):
            e1, e2, e3 = st.columns(3)
            freetime = e1.slider("Free Time (1-5)", 1, 5, vals["freetime"])
            goout = e2.slider("Going Out (1-5)", 1, 5, vals["goout"])
            health = e3.slider("Health Status (1-5)", 1, 5, vals["health"])

            e4, e5 = st.columns(2)
            dalc = e4.slider("Workday Alcohol (1-5)", 1, 5, vals["Dalc"])
            walc = e5.slider("Weekend Alcohol (1-5)", 1, 5, vals["Walc"])

            famrel = 4
            schoolsup = "no"
            famsup = "yes"
            activities = "yes"
            nursery = "yes"
            internet = "yes"
            romantic = "no"
            famsize = "GT3"
            pstatus = "T"
            guardian = "mother"
            reason = "course"

        # Submit button
        submit_btn = st.form_submit_button("🔮 Predict Final Grade", use_container_width=True)

    # When user clicks Predict
    if submit_btn and model is not None:
        # Create input row with all 30 features
        input_data = pd.DataFrame([{
            "school": school, "sex": sex, "age": int(age), "address": address,
            "famsize": famsize, "Pstatus": pstatus, "Medu": int(medu), "Fedu": int(fedu),
            "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
            "traveltime": int(traveltime), "studytime": int(studytime), "failures": int(failures),
            "schoolsup": schoolsup, "famsup": famsup, "paid": paid, "activities": activities,
            "nursery": nursery, "higher": higher, "internet": internet, "romantic": romantic,
            "famrel": int(famrel), "freetime": int(freetime), "goout": int(goout),
            "Dalc": int(dalc), "Walc": int(walc), "health": int(health), "absences": int(absences)
        }])

        raw_pred = float(model.predict(input_data)[0])
        final_grade = max(0.0, min(20.0, raw_pred))

        st.write("---")
        # Display Result nicely
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.metric(label="Predicted Final Grade", value=f"{final_grade:.2f} / 20", delta=f"{final_grade - 10.42:.2f} vs class avg")
        
        with res_col2:
            if final_grade >= 16:
                st.success("🌟 **Distinction (16–20)**\n\nExcellent academic performance expected!")
            elif final_grade >= 14:
                st.info("✨ **Good Standing (14–15)**\n\nConsistent passing performance expected.")
            elif final_grade >= 10:
                st.info("📘 **Passing Grade (10–13)**\n\nMeets the passing threshold.")
            else:
                st.error("⚠️ **At Risk (< 10)**\n\nEarly tutoring and attendance support recommended.")

        st.progress(final_grade / 20.0)

# ----------------------- RIGHT COLUMN: ANALYTICS -----------------------
with col_charts:
    st.subheader("📊 Model & Data Analytics")

    tab1, tab2, tab3 = st.tabs(["🏆 Important Features", "📈 Actual vs Predicted", "📚 Study Time vs Grade"])

    with tab1:
        if df_importance is not None:
            top_features = df_importance.head(10).sort_values("Importance", ascending=True)
            top_features["Feature_Name"] = top_features["Feature"].apply(
                lambda x: x.replace("num__", "").replace("cat__", "").replace("_", " ").title()
            )
            fig1 = px.bar(
                top_features, x="Importance", y="Feature_Name", orientation="h",
                color="Importance", color_continuous_scale="Teal",
                title="Top 10 Features Influencing Final Grade"
            )
            fig1.update_layout(height=350, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Absences, past failures, and parental education are the strongest predictors.")

    with tab2:
        if df_results is not None:
            fig2 = px.scatter(
                df_results, x="Actual Grade", y="RF Predicted Grade",
                color="RF Absolute Error", color_continuous_scale="Viridis",
                title="Test Set Predictions vs Actual Grades (79 Students)"
            )
            fig2.add_shape(type="line", x0=0, y0=0, x1=20, y1=20, line=dict(color="red", dash="dash"))
            fig2.update_layout(height=350, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Red dashed line is the perfect prediction line ($y = x$).")

    with tab3:
        if df is not None:
            fig3 = px.box(
                df, x="studytime", y="G3", color="studytime",
                title="Study Time vs Final Grade Distribution",
                labels={"studytime": "Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", "G3": "Final Grade"}
            )
            fig3.update_layout(height=350, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Students studying more hours per week achieve higher average grades.")

# -----------------------------------------------------------------------------
# 5. FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption("Student Performance Dashboard · Built with Streamlit & Scikit-learn · Author: Tejaswini Rakhunde")
