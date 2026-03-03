import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("Starting Custom Data Pipeline...")
# ---------------------------------------------------------
# 1. Proprietory Data Generation Pipeline
# In a real-world scenario, this would be an SQL query.
# Here, we programmatically construct the dataset to reflect 
# the diverse and imbalanced nature of academic records.
# ---------------------------------------------------------
np.random.seed(42)  # Setting seed ensures reproducibility for our Viva.
n_samples = 1500

math_score = np.random.randint(30, 100, n_samples)
english_score = np.random.randint(30, 100, n_samples)
science_score = np.random.randint(30, 100, n_samples)
internal_marks = np.random.randint(10, 50, n_samples)
attendance = np.random.randint(50, 100, n_samples)
late_submissions = np.random.randint(0, 15, n_samples)
activities = np.random.choice(['Low', 'Medium', 'High'], n_samples)
assignment_completion = np.random.randint(40, 100, n_samples)
study_hours = np.random.randint(2, 30, n_samples)

# Calculate target ('Performance Class') based on strict logic rules.
# This represents the "ground truth" logic from educational psychology:
# Heavily weighting attendance, assignment completion, and core scores.
targets = []
for i in range(n_samples):
    # Composite score formula: 30% Math + 30% Sci + 20% Eng + 20% Internals
    score = 0.3*math_score[i] + 0.3*science_score[i] + 0.2*english_score[i] + 0.2*(internal_marks[i]*2)
    
    # Logic for Classification (The Core Problem Statement)
    if attendance[i] < 70 or score < 50 or late_submissions[i] > 8:
        targets.append(3) # At Risk (Highest priority for early intervention)
    elif score >= 80 and attendance[i] >= 85 and assignment_completion[i] > 80:
        targets.append(0) # Excellent
    elif score >= 65:
        targets.append(1) # Good
    else:
        targets.append(2) # Average

df = pd.DataFrame({
    'Math Score': math_score,
    'English Score': english_score,
    'Science Score': science_score,
    'Internal Assessment': internal_marks,
    'Attendance Percentage': attendance,
    'Late Submission Count': late_submissions,
    'Participation': activities,
    'Assignment Completion Rate': assignment_completion,
    'Study Hours per Week': study_hours,
    'Performance Class': targets
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/student_data.csv', index=False)
print("Data pipeline executed and saved to data/student_data.csv")

# ---------------------------------------------------------
# 2. Data Preprocessing & Feature Engineering
# This is a critical sub-feature. We use ColumnTransformer 
# to prevent data leakage during scaling/encoding.
# ---------------------------------------------------------
X = df.drop('Performance Class', axis=1)  # Features
y = df['Performance Class']               # Target variable

# We separate categorical from numeric for different preprocessing logic.
categorical_cols = ['Participation']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# The Preprocessor logic for our pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # StandardScaler removes the mean and scales to unit variance.
        # This is vital for Gradient Boosting / distance-based algorithms.
        ('num', StandardScaler(), numeric_cols),
        # OneHotEncoder converts categorical text ('Low', 'High') into binary columns 0/1.
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Stratify=y ensures the 80/20 split maintains the same proportion of "At Risk" students in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Models...")
# ---------------------------------------------------------
# 3. Model Training & Evaluation (Methodology)
# We test 3 distinct algorithms to prove our optimization methodology.
# ---------------------------------------------------------
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),          # Baseline: Interpretable but high variance
    'Random Forest': RandomForestClassifier(random_state=42),          # Advanced: Bagging ensemble to reduce overfitting
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)   # Final: Sequential boosting for highest accuracy
}

best_model = None
best_recall = -1
best_model_name = ""

for name, model in models.items():
    # Constructing the pipeline. 
    # This ensures test data is never exposed to the scaler during training (No Data Leakage).
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict on unseen test data
    y_pred = pipeline.predict(X_test)
    
    # Evaluation Metrics calculation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Extracting Recall specifically for Class 3 ("At Risk"). 
    # Why? We want to minimize False Negatives (missing a student who needs help).
    recall_at_risk = report.get('3', {}).get('recall', 0)
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    if '3' in report:
        print(f"Recall for 'At Risk' (Class 3): {recall_at_risk:.4f}\n")
    else:
        print(f"Recall for 'At Risk' (Class 3): N/A (no such samples predictions?)\n")
    
    # OPTIMIZATION STEP: 
    # The model selection prioritized Recall on Class 3 over raw Accuracy.
    if recall_at_risk > best_recall:
        best_recall = recall_at_risk
        best_model = pipeline
        best_model_name = name

print(f"Best Model based on 'At Risk' recall is: {best_model_name}")

# ---------------------------------------------------------
# 4. Serialize (Save) the Model
# We save the pipeline (which includes the Scaler and Encoder) 
# so the Streamlit app can directly use the .predict() method on raw inputs.
# ---------------------------------------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/student_performance_model.joblib')
print("Model saved to models/student_performance_model.joblib")
