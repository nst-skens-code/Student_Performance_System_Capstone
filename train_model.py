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

print("Starting custom data generation...")

# set seed for reproducible results
np.random.seed(42)  
n_samples = 1500

# Generating features for the students
math_score = np.random.randint(30, 100, n_samples)
english_score = np.random.randint(30, 100, n_samples)
science_score = np.random.randint(30, 100, n_samples)
internal_marks = np.random.randint(10, 50, n_samples)
attendance = np.random.randint(50, 100, n_samples)
late_submissions = np.random.randint(0, 15, n_samples)
activities = np.random.choice(['Low', 'Medium', 'High'], n_samples)
assignment_completion = np.random.randint(40, 100, n_samples)
study_hours = np.random.randint(2, 30, n_samples)

# calculate the target class based on some rules we defined
targets = []
for i in range(n_samples):
    # calculate a weighted score
    score = 0.3*math_score[i] + 0.3*science_score[i] + 0.2*english_score[i] + 0.2*(internal_marks[i]*2)
    
    # assign classes (0=Excellent, 1=Good, 2=Average, 3=At Risk)
    if attendance[i] < 70 or score < 50 or late_submissions[i] > 8:
        targets.append(3) 
    elif score >= 80 and attendance[i] >= 85 and assignment_completion[i] > 80:
        targets.append(0) 
    elif score >= 65:
        targets.append(1) 
    else:
        targets.append(2) 

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
print("Data saved to data/student_data.csv")

# ==========================================
# Data Preprocessing
# ==========================================

X = df.drop('Performance Class', axis=1) 
y = df['Performance Class']              

# Setup preprocessing for categorical vs numeric data
categorical_cols = ['Participation']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining models...")

# dict of models we want to test
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),          
    'Random Forest': RandomForestClassifier(random_state=42),          
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)   
}

best_model = None
best_recall = -1
best_model_name = ""

for name, model in models.items():
    # create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # extract recall for class 3 since that's our focus
    recall_at_risk = report.get('3', {}).get('recall', 0)
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    if '3' in report:
        print(f"Recall for 'At Risk' (Class 3): {recall_at_risk:.4f}\n")
    else:
        print(f"Recall for 'At Risk' (Class 3): N/A\n")
    
    # keep track of best model based on recall
    if recall_at_risk > best_recall:
        best_recall = recall_at_risk
        best_model = pipeline
        best_model_name = name

print(f"Best Model found: {best_model_name}")

# save it for the streamlit app
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/student_performance_model.joblib')
print("Model saved to models/student_performance_model.joblib")
