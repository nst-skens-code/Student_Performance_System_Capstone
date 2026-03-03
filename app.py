import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# Ensure model exists before trying to load it
@st.cache_resource
def load_model():
    model_path = 'models/student_performance_model.joblib'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

class_mapping = {
    0: ("Excellent", "High marks and consistent performance", "green"),
    1: ("Good", "Above average academic performance", "blue"),
    2: ("Average", "Moderate performance", "orange"),
    3: ("At Risk", "Low marks and poor attendance", "red")
}

st.title("🎓 Student Performance Prediction System")
st.markdown("Predict the performance category of a student based on their academic and behavioral metrics. Identifying **At Risk** students allows for early academic intervention.")

if model is None:
    st.warning("Model not found! Please run `python train_model.py` to generate the model first.")
    st.stop()

# Create layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Academic Metrics")
    math_score = st.number_input("Math Score (0-100)", min_value=0, max_value=100, value=75)
    english_score = st.number_input("English Score (0-100)", min_value=0, max_value=100, value=75)
    science_score = st.number_input("Science Score (0-100)", min_value=0, max_value=100, value=75)
    internal_marks = st.number_input("Internal Assessment (0-50)", min_value=0, max_value=50, value=35)

with col2:
    st.subheader("📅 Attendance")
    attendance = st.number_input("Attendance Percentage (0-100)", min_value=0, max_value=100, value=85)
    late_submissions = st.number_input("Late Submission Count", min_value=0, max_value=20, value=2)

with col3:
    st.subheader("🧠 Behavioral")
    activities = st.selectbox("Participation in Activities", options=['Low', 'Medium', 'High'], index=1)
    assignment_completion = st.number_input("Assignment Completion Rate (%)", min_value=0, max_value=100, value=80)
    study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=50, value=10)

st.markdown("---")

if st.button("Predict Performance", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        'Math Score': [math_score],
        'English Score': [english_score],
        'Science Score': [science_score],
        'Internal Assessment': [internal_marks],
        'Attendance Percentage': [attendance],
        'Late Submission Count': [late_submissions],
        'Participation': [activities],
        'Assignment Completion Rate': [assignment_completion],
        'Study Hours per Week': [study_hours]
    })
    
    # Predict
    prediction = model.predict(input_data)[0]
    category, description, color = class_mapping[prediction]
    
    st.subheader("Prediction Result")
    
    # Render custom styled alert based on prediction
    if category == "At Risk":
        st.error(f"### 🛑 {category}\n**{description}**\n\n*Immediate intervention recommended.*")
    elif category == "Excellent":
        st.success(f"### 🌟 {category}\n**{description}**")
    elif category == "Good":
        st.info(f"### 👍 {category}\n**{description}**")
    else:
        st.warning(f"### ⚠️ {category}\n**{description}**")
    
    # Simple feature overview plot
    st.markdown("#### Input Metrics Overview")
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Math", "English", "Science", "Attendance", "Assignments"],
            "Score": [math_score, english_score, science_score, attendance, assignment_completion]
        }
    )
    st.bar_chart(metrics_df.set_index("Metric"), height=200, color=color)
