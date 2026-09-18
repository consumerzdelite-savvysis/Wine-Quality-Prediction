# Wine Quality Prediction

An end-to-end machine learning project for predicting wine quality from physicochemical properties, with a deployed FastAPI backend and Streamlit web application.

---

## Project Overview

Wine quality is influenced by several physicochemical properties, including acidity, residual sugar, chlorides, sulphates, sulphur dioxide, density, and alcohol content. Understanding how these properties relate to wine quality can support more consistent and data-driven quality assessment.

This project develops an end-to-end Machine Learning solution for predicting wine quality from measurable physicochemical properties.

The project covers the complete Machine Learning workflow:

- Data collection and preparation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing
- Model development
- Model comparison
- Hyperparameter tuning
- Prediction
- Model evaluation
- FastAPI backend development
- Streamlit frontend (Web application) development
- Cloud deployment
- End-to-end validation
- Documentation

The completed solution allows users to enter wine physicochemical properties through a web interface and receive a predicted wine quality score.

---

## Project Objective

The objective of this project is to develop a machine learning model capable of predicting wine quality from measurable physicochemical properties.

The project also demonstrates how a Machine Learning model can be transformed from a development notebook into an accessible web-based production system.

---

## Problem Statement

Traditional wine quality assessment can depend heavily on human sensory evaluation and expert judgment. Although these approaches are valuable, they still can be subjective and time-consuming.

This project explores how Machine Learning can use measurable chemical characteristics of wine to provide a consistent, data-driven prediction of wine quality.

---

## Why This Project Matters

A wine quality prediction system can provide useful decision-support information for:

- Wine producers and quality-control teams
- Food and beverage businesses
- Researchers and data scientists
- Product development teams
- Quality-assurance processes

The broader goal is to demonstrate how Machine Learning can transform structured scientific data into practical predictive insights.

---

## Dataset

The project uses the **Wine Quality Dataset** containing physicochemical measurements of red and white Portuguese wines.

The dataset originates from the **UCI Machine Learning Repository** and is also available through Kaggle.

The original combined dataset contained:

- 6,497 observations
- 11 original physicochemical predictor variables
- 1 derived predictor variable (`wine_type`)
- 1 target variable (`quality`)

After removing exact duplicate records:
- 5,320 observations remained

Exact duplicate removal was performed to reduce redundancy and potential bias during model development.

## Dataset Files

The raw datasets are stored in:

```text
data/
├── raw/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── external/
│   └── Wine Quality Dataset.csv
└── processed/
```
---

## Main Features

The model uses the following 12 predictor variables:

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Wine type

The `wine_type` feature is encoded as:

- Red = 0
- White = 1

The target variable is:

**Quality** — the wine quality score.

## Project Workflow

The project follows an end-to-end Machine Learning workflow:

Raw Data
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Data Preprocessing & Scaling
   ↓
Model Development
 ↓
Model Comparison
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Prediction
   ↓
FastAPI Backend Development
   ↓
Streamlit Frontend Development
   ↓
Cloud Deployment
   ↓
End-to-End Validation

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure, quality, distribution, and relationships within the dataset.

The analysis included:

- Dataset structure
- Data types
- Missing-value inspection
- Duplicate-value detection
- Descriptive statistics
- Distribution analysis
- Outlier analysis
- Relationship between variables
- Correlation analysis
- Relationships between physicochemical properties and wine quality

One of the major findings from the correlation analysis was that **alcohol showed the strongest positive relationship with wine quality**, while density showed a weak negative relationship with quality.

The correlation analysis also showed that some variables had weak linear relationships with quality. However, a weak linear relationship does not necessarily mean that a feature is unsuitable for machine learning because nonlinear models can identify relationships that correlation analysis does not capture.

## Data Preparation and Preprocessing

Data preparation and preprocessing were performed to prepare the dataset for Machine Learning.

The workflow included:
- Combining the red and white wine datasets
- Creating the `wine_type` feature
- Cleaning the dataset
- Handling duplicate observations
- Preparing the target variable
- Separating features from the target
- Splitting the data into training and testing sets
- Scaling numerical features where required
- Preparing the final feature matrix for model training

The original model-development experiment used an 80/20 train-test split with `random_state=42`.

The dataset was divided into:
-	**Training set**: 4,256 samples
-	**Testing set**: 1,064 samples

A fitted StandardScaler was saved and reused during prediction so that incoming data is transformed consistently with the data used to train the model.

## Machine Learning Model Development

Several machine learning classification algorithms were evaluated during the original model-development experiment:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Multi-Layer Perceptron (MLP)

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

Random Forest was included because it is well suited to structured tabular data and can capture nonlinear relationships between physicochemical properties and wine quality.

The trained model is stored in:
models/random_forest_model.pkl

The fitted preprocessing scaler is stored in:
models/scaler.pkl

These saved artifacts allow the deployed application to make predictions without retraining the model whenever the application starts.

## Model Evaluation and Selection

Random Forest achieved the highest **accuracy, precision, and recall** among the compared models and was selected as the final deployed model. LightGBM achieved a marginally higher **F1 Score**, which is documented transparently below.

### Original Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 54.42% | 55.43% | 54.42% | 51.44% |
| Decision Tree | 45.11% | 46.88% | 45.11% | 45.53% |
| **Random Forest** | **57.71%** | **58.83%** | **57.71%** | **55.61%** |
| KNN | 52.73% | 51.71% | 52.73% | 51.43% |
| SVM | 55.83% | 58.30% | 55.83% | 52.24% |
| XGBoost | 56.20% | 55.51% | 56.20% | 54.20% |
| LightGBM | 57.14% | 57.22% | 57.14% | **55.80%** |
| MLP | 55.26% | 55.83% | 55.26% | 52.71% |

Random Forest achieved:

- Accuracy: 57.71%
- Precision: 58.83%
- Recall: 57.71%
- F1 Score: 55.61%

LightGBM achieved a marginally higher F1 score of 55.80%, but Random Forest achieved the highest accuracy, precision, and recall.

Based on the project's primary evaluation criteria, Random Forest was therefore selected as the final model.

### Hyperparameter Tuning Experiment

A separate notebook, `notebooks/Hyperparameter_Tuning.ipynb`, was created to investigate whether the baseline Random Forest could be improved through hyperparameter optimization.

The original model-development notebook was intentionally left unchanged so that the original experimental records remain unchanged.

The tuning experiment used:

- 5-fold Stratified K-Fold cross-validation
- RandomizedSearchCV
- 30 parameter combinations
- Random Forest hyperparameter optimization

The best parameter configuration identified during the tuning was:

```text
bootstrap = True
max_depth = 30
max_features = sqrt
min_samples_leaf = 3
min_samples_split = 9
n_estimators = 288
```

It achieved a cross-validation accuracy of approximately 57.07%.

The tuned Random Forest model achieved:

- Accuracy: 57.33%
- Precision: 55.11%
- Recall: 57.33%
- F1 Score: 54.20%

The tuned model therefore did not outperform the original baseline Random Forest, which achieved 57.71% accuracy. Consequently, the original baseline Random Forest was retained as the final deployed model.

The hyperparameter-tuning experiment is intentionally maintained separately from the original model-development notebook so that the original experimental record remains unchanged.

## Prediction

The trained model successfully produces wine quality predictions from supplied physicochemical properties.

During system testing, the prediction pipeline successfully returned:
**Predicted Wine Quality: 6**

The prediction was successfully validated locally and through the deployed cloud-based API (FastAPI) and Streamlit application.

## Backend — FastAPI
The project includes a FastAPI backend responsible for:
- Receiving wine physicochemical properties.
- Validating incoming data.
- Applying the saved StandardScaler.
- Passing the transformed data to the Random Forest model.
- Returning the predicted wine quality.

The backend is located in:
backend/main.py

**Backend Technologies**
•	FastAPI
•	Pydantic
•	Joblib
•	NumPy
•	Pandas
•	Scikit-learn
•	Uvicorn

The API provides the interface through which the frontend communicates with the trained Machine Learning model.

## Live FastAPI Backend

**FastAPI Backend**:
https://wine-quality-prediction-backend.onrender.com⁠

**Swagger API Documentation**
Interactive Swagger documentation is available at:
https://wine-quality-prediction-backend.onrender.com/docs⁠

The deployed /predict endpoint was successfully tested remotely and returned:

**Predicted Wine Quality: 6**

## Frontend — Streamlit

A Streamlit frontend was developed to provide a simple user interface for submitting wine physicochemical properties and receiving a predicted quality score.

The frontend is located in:
frontend/app.py

The Streamlit application communicates with the FastAPI backend through HTTP requests.

## Live Streamlit Application
The deployed Streamlit application is available at:
https://wine-quality-prediction-frontend-x4ho.onrender.com⁠

The public application was successfully tested after deployment and returned:

**Predicted Wine Quality: 6**

## API–Frontend Integration
The Streamlit frontend communicates with the deployed FastAPI backend through the /predict endpoint.

The production architecture is:

User
  ↓
Streamlit Frontend
  ↓
HTTP Request
  ↓
FastAPI Backend
  ↓
Preprocessing / StandardScaler
  ↓
Random Forest Model
  ↓
Wine Quality Prediction
  ↓
Streamlit Frontend
  ↓
User

The integration was tested locally and remotely.

The final end-to-end validation successfully returned:

**Predicted Wine Quality: 6**

## Cloud Deployment

The application was deployed to the cloud using **Render**.

## Backend Deployment
The FastAPI backend was deployed as a Render Web Service.

**Live Backend**:
https://wine-quality-prediction-backend.onrender.com⁠

**Swagger  API Documentation**:
https://wine-quality-prediction-backend.onrender.com/docs⁠

## Frontend Deployment
The Streamlit frontend was deployed as a separate Render Web Service.

**Live Application**:
https://wine-quality-prediction-frontend-x4ho.onrender.com⁠

## Deployment Architecture

                    Internet User
                         │
                         ▼
              ┌─────────────────────┐
              │ Streamlit Frontend  │
              │       Render        │
              └──────────┬──────────┘
                         │
                         │ HTTPS
                         ▼
              ┌─────────────────────┐
              │   FastAPI Backend   │
              │       Render        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  StandardScaler +   │
              │  Random Forest      │
              │      Model          │
              └──────────┬──────────┘
                         │
                         ▼
                 Wine Quality Score

## Project Structure

Wine-Quality-Prediction/
│
├── backend/
│   └── main.py
│
├── data/
│   ├── external/
│   │   └── Wine Quality Dataset.csv
│   ├── processed/
│   └── raw/
│       ├── winequality-red.csv
│       └── winequality-white.csv
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── Hyperparameter_Tuning.ipynb
│   └── Wine_Quality_Prediction.ipynb
│
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── predictions.csv
│
├── screenshots/
│   ├── 01_eda_overview.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_model_comparison.png
│   ├── 04_final_model_ranking.png
│   ├── 05_fastapi_backend.png
│   ├── 06_fastapi_swagger.png
│   ├── 07_fastapi_prediction.png
│   ├── 08_streamlit_frontend.png
│   ├── 09_render_backend.png
│   ├── 10_render_frontend.png
│   └── 11_final_prediction.png
│
├── src/
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt

The *src* directory contains supporting project scripts. The primary model-development and training workflow is documented in notebooks/Wine_Quality_Prediction.ipynb.

## Technologies Used
-  VS Code
-	Python
-	Pandas
-	NumPy
-	Matplotlib
-	Seaborn
-	Scikit-learn
-	Joblib
-	FastAPI
-	Uvicorn
-	Pydantic
-	Streamlit
-	Git
-	GitHub
-	Render

## Installation and Local Setup

**1. Clone the Repository**
git clone https://github.com/consumerzdelite-savvysis/Wine-Quality-Prediction.git

**2. Create a Virtual Environment**
python -m venv .venv

**3. Activate the Virtual Environment on Windows**
.venv\Scripts\Activate.ps1

**4. Install Dependencies**
pip install -r requirements.txt

## Running the Backend Locally

From the project root:
uvicorn backend.main:app --reload

The FastAPI backend will normally be available at:
http://127.0.0.1:8000

Interactive API documentation:
http://127.0.0.1:8000/docs

## Running the Frontend Locally

Open another terminal, activate the virtual environment, and run:
streamlit run frontend/app.py

Streamlit will provide a local URL that can be opened in a web browser.

## Model Files

The trained model and preprocessing scaler are included in the project:

models/
├── random_forest_model.pkl
└── scaler.pkl

These files allow the prediction application to use the trained model and fitted preprocessing scaler without retraining the model every time the application starts.

## Reproducibility

The dependencies required by the application and supporting project scripts are recorded in requirements.txt. The notebooks also document additional model-comparison experiments conducted during development.

The .gitignore file prevents environment-specific and unnecessary files such as .venv, Python cache files, notebook checkpoints, and temporary files from being committed to GitHub.

## Project Reports

Model evaluation results and prediction outputs are stored in:

reports/
├── metrics.json
└── predictions.csv

The `reports/figures/` directory is reserved for report-specific visualizations. Visual evidence from the completed project is
currently maintained in the `screenshots/` directory.

## API
The machine learning model is exposed through a FastAPI backend.
Local API
http://127.0.0.1:8000

Prediction Endpoint
POST /predict

Swagger Documentation
/docs

## Screenshots

The following screenshots provide visual evidence of the project's development, analytical workflow, model evaluation, API implementation, frontend interface, cloud deployment, and final prediction.

### 1. Exploratory Data Analysis — Summary Statistics

![EDA Overview](screenshots/01_eda_overview.png)

The summary statistics provide an overview of the distribution and descriptive characteristics of the wine-quality dataset.

### 2. Correlation Heatmap

![Correlation Heatmap](screenshots/02_correlation_heatmap.png)

The correlation analysis shows the relationships between the physicochemical properties and wine quality. Alcohol showed the strongest positive relationship with wine quality, while density showed a weak negative relationship.

### 3. Model Comparison

![Model Comparison](screenshots/03_model_comparison.png)

This comparison presents the evaluation results of the machine learning models considered during the project.

### 4. Final Model Ranking

![Final Model Ranking](screenshots/04_final_model_ranking.png)

The final model ranking provides a comparative view of model performance and supports the selection of the final prediction model.

### 5. FastAPI Backend

![FastAPI Backend](screenshots/05_fastapi_backend.png)

The FastAPI backend provides the API service responsible for receiving wine-property inputs and returning predictions from the trained machine learning model.

### 6. FastAPI Swagger Documentation

![FastAPI Swagger](screenshots/06_fastapi_swagger.png)

The Swagger interface provides interactive documentation and testing for the deployed FastAPI endpoints.

### 7. FastAPI Prediction

![FastAPI Prediction](screenshots/07_fastapi_prediction.png)

The deployed API successfully processed a prediction request and returned a wine quality prediction.

### 8. Streamlit Frontend

![Streamlit Frontend](screenshots/08_streamlit_frontend.png)

The Streamlit frontend provides a user-friendly interface for entering wine physicochemical properties and obtaining a predicted quality score.

### 9. Render Backend Deployment

![Render Backend](screenshots/09_render_backend.png)

The FastAPI backend was successfully deployed as a live web service on Render.

### 10. Render Frontend Deployment

![Render Frontend](screenshots/10_render_frontend.png)

The Streamlit frontend was successfully deployed as a live web service on Render.

### 11. Final Prediction

![Final Prediction](screenshots/11_final_prediction.png)

The deployed application successfully returned:

**Predicted Wine Quality: 6**

## Limitations

The model has moderate predictive performance and should not be interpreted as a replacement for professional wine-quality assessment.

Important limitations include:
* Wine quality is subjective and may depend on human sensory assessment.
* The dataset represents a specific wine-producing context.
* The target classes are imbalanced.
* The classification formulation treats quality scores as discrete classes and does not explicitly model their ordinal nature.
* Additional domain-specific features could potentially improve performance.

## Future Improvements

Potential future improvements include:

- Additional feature engineering
- More extensive hyperparameter optimization
- Class-imbalance handling techniques
- Ordinal classification approaches
- Regression-based modelling
- Explainable AI and enhanced feature-importance analysis
- More extensive cross-validation
- Larger and more diverse datasets
- Automated model retraining
- Production monitoring
- Continuous integration and deployment
- Further frontend improvements

## Key Business Insight

The analysis indicates that alcohol content has the strongest positive linear relationship with the recorded wine quality score among the measured variables.

However, wine quality is influenced by multiple physicochemical characteristics rather than a single variable. A machine learning model can therefore support quality-screening and decision-support processes by identifying likely quality levels from measurable properties.

Potential business applications include:

* Production quality screening
* Batch monitoring
* Quality-control support
* Wine classification
* Data-driven production analysis
* Decision support for wineries

The model should be treated as a decision-support tool rather than an autonomous quality-certification system.

## Final Deliverables

The completed project provides:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Machine learning model comparison
* Random Forest classification model
* Model evaluation
* Prediction
* Hyperparameter-tuning experiment
* Saved model and scaler
* FastAPI prediction API
* Swagger API documentation
* Streamlit user interface
* Cloud deployment
* GitHub repository
* Academic project documentation
* Project screenshots

## Author

**Name**: Segun Daramola

**Cohort**: Data Science / AI and Machine Learning Cohort 3

**Programme**: 3MTT/DSN/DeepTech/WesOnline Mentorship
---

## Mentor / Training Credit

*Godspower Uyanga*
**GworldSoft Solutions Limited / 3MTT (DSN/DeepTech_Ready/WesOnline)**
---

## Project Status

**Current Status**: Completed end-to-end Machine Learning solution with cloud deployment, successful remote validation, and supporting documentation.

The project has been developed as an end-to-end machine learning application, from data preparation and exploratory analysis through model development, evaluation, API development, frontend development, deployment, and documentation.

## Final Validation

The deployed application was successfully tested end-to-end.

Final result:

**Predicted Wine Quality: 6**

This confirms that the trained model, saved preprocessing scaler, FastAPI backend, Streamlit frontend, and cloud deployment are working together as an end-to-end prediction system.

## Live Application

**Streamlit Frontend**:
https://wine-quality-prediction-frontend-x4ho.onrender.com⁠

**FastAPI Backend**:
https://wine-quality-prediction-backend.onrender.com⁠

**FastAPI Swagger Documentation**:
https://wine-quality-prediction-backend.onrender.com/docs⁠

## Repository

The complete project source code, machine learning notebook, model artifacts, reports, screenshots, API, frontend, and documentation are maintained on GitHub:

https://github.com/consumerzdelite-savvysis/Wine-Quality-Prediction
