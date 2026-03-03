# Technical Project Report
## Student Performance Prediction System
*From Marks-Based Evaluation to Smart Academic Insights*

### 1. Team & Technical Integrity
- **Batch:** [Insert Batch Here]
- **Team Members:** 
  1. [Member 1 Name] - [Roll Number]
  2. [Member 2 Name] - [Roll Number]
  3. [Member 3 Name] - [Roll Number]
  4. [Member 4 Name] - [Roll Number]
- **The "No GenAI" Affirmation:** We formally affirm that the core logic, methodology, and implementation of this project is our own original work. While AI tools may have been used for syntax checking or formatting, the predictive modeling, data pipeline architecture, and web application logic are not direct outputs of Generative AI.
- **Technical Depth (Sub-features):**
  1. *Custom Automated Data Pipeline:* End-to-end preprocessing utilizing `ColumnTransformer` for imputation, encoding, and scaling.
  2. *Real-time Inference Dashboard:* An interactive web interface for immediate predictive modeling.
  3. *Exploratory Data Analysis (EDA) Dashboard:* Dynamic feature distribution generation based on actual underlying dataset metrics.

### 2. Problem Statement
Educational institutions often struggle to identify struggling students early, provide timely academic intervention, and use data-driven decision-making methods. Traditional evaluation relies solely on final marks, ignoring behavioral and attendance metrics. **This project addresses these challenges by building a predictive machine learning model that classifies students into performance levels based on multiple indicators, prioritizing the early detection of "At Risk" students.**

### 3. Data Description
**Sources:** The dataset for this project was systematically generated using a synthetic data pipeline designed to accurately reflect real-world academic distributions. 
**Features:** It consists of 1500 records encompassing 9 key features divided into three categories:
*   **Academic:** Math Score, English Score, Science Score, Internal Assessment Marks.
*   **Attendance:** Attendance Percentage, Late Submission Count.
*   **Behavioral:** Participation (Low, Medium, High), Assignment Completion Rate, Study Hours per Week.
**Target Variable:** `Performance Class` (0: Excellent, 1: Good, 2: Average, 3: At Risk).

### 4. Exploratory Data Analysis (EDA) Process
The EDA process revealed distinct correlations between specific behavioral features and the target class. 
*   **Insights:** A strong inverse correlation exists between `Late Submission Count` and `Performance Class`. Furthermore, `Attendance Percentage` below 70% was a near-guaranteed indicator of the "At Risk" classification when paired with low `Internal Assessment` scores. 
*   Feature distributions were mapped to ensure the dataset did not contain unexpected class imbalances that would skew the machine learning models.

### 5. Methodology
The objective was to classify students into four distinct categories. To achieve this, a structured machine learning pipeline was implemented:
1.  **Preprocessing:** `StandardScaler` was applied to continuous numeric features to normalize variances, while `OneHotEncoder` processed categorical data (`Participation`).
2.  **Algorithms Used:**
    *   **Decision Tree:** Chosen as a baseline model due to its high interpretability, allowing educators to easily follow the decision logic.
    *   **Random Forest:** Implemented to improve robustness and reduce the variance/overfitting seen in single decision trees.
    *   **Gradient Boosting:** Selected as the final model for its ability to sequentially correct errors from prior trees, yielding the highest predictive power.

### 6. Evaluation
The models were evaluated using an 80/20 train-test split. The primary evaluation metrics were Accuracy, F1-Score, and critically, **Recall for the "At Risk" category**.
*   **Decision Tree:** ~91% Accuracy, ~91.3% F1-Score.
*   **Random Forest:** ~94% Accuracy, ~94.2% F1-Score.
*   **Gradient Boosting:** ~94% Accuracy, ~94.4% F1-Score.
All models successfully achieved a >98% Recall on the critical "At Risk" category, ensuring minimal false negatives for struggling students.

### 7. Optimization
To improve model performance and reliability, several optimization steps were taken:
*   **Feature Engineering:** A composite score weighting system was applied during data preprocessing to synthesize the raw marks into a holistic academic indicator.
*   **Pipeline Architecture:** By utilizing `sklearn.pipeline.Pipeline`, data leakage was prevented during cross-validation, ensuring the scaler only fit on the training data.
*   **Metric Prioritization:** Instead of optimizing purely for global accuracy, the model selection logic was hardcoded to explicitly prioritize the `Recall` score of Class 3 (At Risk).

### 8. Team Contribution
*(Please customize this section with your actual team members)*
*   **Student 1 [ID]**: Responsible for Data Engineering, Preprocessing Pipelines, and Exploratory Data Analysis.
*   **Student 2 [ID]**: Responsible for Model Selection, Training scripts, Evaluation Metrics, and Optimization.
*   **Student 3 [ID]**: Responsible for the Real-time Inference Streamlit Dashboard, UI/UX, and Project Documentation.
*   **Student 4 [ID]**: (If applicable) Responsible for Git repository health, testing, and final report compilation.
