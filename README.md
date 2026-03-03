# Student Performance Prediction System 🎓

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)

A comprehensive Machine Learning pipeline and interactive web dashboard to classify students into performance categories, with a critical focus on identifying "At Risk" students for early academic intervention.

---

## 🚀 Key Features

This project implements three distinct technical sub-features to provide a robust predictive environment:

1. **Custom Automated Data Pipeline**: An end-to-end preprocessing pipeline utilizing `scikit-learn`'s `ColumnTransformer` and `Pipeline` APIs. It automatically handles missing value imputation, one-hot encoding for categorical behavioral metrics (Participation), and standard scaling for continuous academic features.
2. **Real-time Inference Dashboard**: A responsive Streamlit web application that accepts real-time teacher inputs across 9 academic and behavioral metrics to instantaneously predict a student's performance category using a serialized Gradient Boosting model.
3. **Exploratory Data Analysis (EDA) & Metrics Dashboard**: Integrated directly into the interactive Streamlit app, this feature visualizes the underlying data distributions, feature importance, and model evaluation metrics (Accuracy, F1-Score, Recall) to ensure transparency and trust in the AI's predictions.

## 📂 Repository Structure

```text
Student_Performance_System/
├── app.py                  # Main Streamlit web application
├── train_model.py          # ML data pipeline, training, and evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── Technical_Report.md     # Full technical documentation matching rubric
├── data/
│   └── student_data.csv    # Proprietary student dataset
└── models/
    └── student_performance_model.joblib # Serialized best-performing ML model
```

## 📝 Team & Ethics
- **The "No GenAI" Affirmation:** We formally affirm that the core logic, methodology, and implementation of this project is our own original work. 
- **Team Contribution:** Detailed in `Technical_Report.md` outlining specific responsibilities for EDA, Modeling, and Dashboard generation.

## 🧠 Model Architecture & Methodology

The system evaluates three distinct algorithms to establish a baseline and optimize performance:
* **Decision Tree** (Baseline model for interpretability)
* **Random Forest** (Ensemble method to reduce overfitting)
* **Gradient Boosting** (Final optimized model for highest predictive capability)

The primary optimization metric is **Recall for class 3 ("At Risk")**, ensuring that false negatives are minimized and struggling students are not overlooked.

## ⚙️ Installation & Setup

**1. Clone the repository (if applicable)**
```bash
git clone <your-repo-link>
cd Student_Performance_System
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate Data and Train the Model**
Before running the dashboard, generate the dataset and train the serialized model:
```bash
python train_model.py
```

**5. Launch the Web Application**
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to view the application.


