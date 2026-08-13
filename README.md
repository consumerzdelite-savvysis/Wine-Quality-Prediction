## Wine Quality Prediction

## Project Overview

Wine quality is influenced by several physicochemical properties, including acidity, residual sugar, chlorides, sulphates, sulphur dioxide, density, and alcohol content. Understanding how these properties relate to wine quality can support more consistent and data-driven quality assessment.

This project develops an end-to-end Machine Learning solution for predicting wine quality from physicochemical properties. The project covers data preparation, exploratory data analysis, feature engineering, machine learning model development, model evaluation, prediction, and deployment through a FastAPI backend and Streamlit frontend.

## Project Objective

The objective of this project is to develop a machine learning model capable of predicting wine quality from measurable physicochemical properties.

The project also demonstrates the complete machine learning workflow, from raw data preparation through model development and evaluation to deployment as an accessible application.

## Problem Statement

Traditional wine quality assessment can depend heavily on human sensory evaluation and expert judgment. Although these approaches are valuable, they can be subjective and time-consuming.

This project explores how machine learning can use measurable chemical characteristics of wine to predict its quality score and provide a consistent, data-driven assessment.

## Why This Project Matters

A reliable wine quality prediction system can provide useful decision-support information for:

* Wine producers and quality-control teams
* Food and beverage businesses
* Researchers and data scientists
* Product development teams
* Quality-assurance processes

The broader goal is to demonstrate how machine learning can transform structured scientific data into practical predictive insights.

## Dataset

The project uses the Wine Quality Dataset containing physicochemical measurements of red and white Portuguese wines.

## Dataset sources

The dataset is associated with the **kaggle.com Machine Learning Repository** and is also available through Kaggle.

The project contains the raw datasets in:

data/raw/
├── winequality-red.csv
└── winequality-white.csv

An additional dataset file is stored under:

data/external/
└── Wine Quality Dataset.csv

## Main features

The dataset contains physicochemical variables including:

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol

The target variable is:
*Quality* - the wine quality score.

## Project Workflow

The project follows a complete machine learning workflow:

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
Data Preprocessing
   ↓
Model Development
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Prediction
   ↓
FastAPI Backend
   ↓
Streamlit Frontend
   ↓
Cloud Deployment

**Exploratory Data Analysis**

Exploratory Data Analysis was performed to understand the structure, quality, distribution, and relationships within the dataset.

The analysis included:

* Dataset structure and dimensions
* Missing-value inspection
* Duplicate-value detection
* Descriptive statistics
* Distribution analysis
* Outlier analysis
* Correlation analysis
* Relationship between physicochemical properties and wine quality

The combined dataset initially contained **6,497 records and 13 columns**.

After duplicate removal, the dataset contained **5,320 records**.

A major finding from the correlation analysis was that **alcohol showed the strongest positive relationship with wine quality**, while density showed a weak negative relationship with quality.

## Feature Engineering and Preprocessing

Feature engineering and preprocessing were performed to prepare the dataset for machine learning.

The workflow included:

* Cleaning the dataset
* Handling duplicate observations
* Preparing the target variable
* Separating features from the target
* Splitting the data into training and testing sets
* Scaling numerical features where required
* Preparing the final feature matrix for model training

The dataset was divided into:

* **Training set**: 4,256 samples
* **Testing set**: 1,064 samples

## Machine Learning Model

The project uses a Random Forest machine learning model for wine quality prediction.

Random Forest was selected because it is well suited to structured tabular data and can capture nonlinear relationships between physicochemical properties and wine quality.

The trained model is stored in:
models/random_forest_model.pkl

The fitted scaler is stored in:
models/scaler.pkl

## Prediction

The trained model successfully produced wine quality predictions from supplied physicochemical properties.

During application testing, the prediction system successfully returned:

*Predicted Wine Quality: 6*

This confirmed that the prediction pipeline was functioning correctly during local integration testing.

## Backend — FastAPI

The project includes a FastAPI backend responsible for receiving wine-property inputs and returning machine learning predictions.

The backend is located in:
backend/main.py

The backend uses:

* FastAPI
* Pydantic
* Joblib
* NumPy

The API provides the interface through which the frontend communicates with the trained machine learning model.

## Frontend — Streamlit

A Streamlit frontend was developed to provide a simple user interface for submitting wine physicochemical properties and receiving a predicted quality score.

The frontend is located in:
frontend/app.py

The Streamlit application communicates with the FastAPI backend using HTTP requests.

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
│   └── Wine_Quality_Prediction.ipynb
│
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── predictions.csv
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

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Pydantic
* Streamlit
* Git
* GitHub

## Installation and Local Setup

## 1. Clone the repository
git clone https://github.com/consumerzdelite-savvysis/Wine-Quality-Prediction.git
cd Wine-Quality-Prediction

## 2. Create a virtual environment
python -m venv .venv

## 3. Activate the virtual environment on Windows
.venv\Scripts\Activate.ps1
## 4. Install dependencies
pip install -r requirements.txt

## Running the Backend

From the project root:
uvicorn backend.main:app --reload

The FastAPI backend will normally be available locally at:
http://127.0.0.1:8000

The interactive API documentation can be accessed at:
http://127.0.0.1:8000/docs

## Running the Frontend

Open another terminal, activate the virtual environment, and run:

streamlit run frontend/app.py

Streamlit provides a local URL that can be opened in a web browser.

## API–Frontend Integration

The Streamlit frontend communicates with the FastAPI backend through HTTP requests.

The integration was tested locally using wine physicochemical-property inputs.

The system successfully processed the input through the trained Random Forest model and returned:

*Predicted Wine Quality: 6*

Cloud deployment replaces the local backend address with the deployed API URL.

## Deployment

The deployment stage makes the machine learning application accessible through the internet.

The intended architecture is:

User
 ↓
Streamlit Frontend
 ↓
FastAPI Backend
 ↓
Random Forest Model
 ↓
Wine Quality Prediction

## Deployment targets

The backend can be deployed using a platform such as:

* Render
* Railway
* Hugging Face Spaces

The Streamlit frontend can be deployed using:

* Streamlit Community Cloud
* Hugging Face Spaces

## Live Demo

Frontend: To be added after deployment.

Backend API: To be added after deployment.

Model Files
The trained model and preprocessing scaler are included in the project:

models/random_forest_model.pkl
models/scaler.pkl
These files allow the prediction application to use the trained model without retraining it every time the application starts.

Reproducibility
The project dependencies are recorded in:

requirements.txt
This allows another user or developer to recreate the Python environment required to run the project.

The .gitignore file prevents environment-specific and unnecessary files such as .venv, Python cache files, notebook checkpoints, and temporary files from being committed to GitHub.

Project Reports
Model evaluation results and prediction outputs are stored in:

reports/
├── metrics.json
└── predictions.csv
Visualizations generated during the project are stored in:

reports/figures/
Screenshots
Screenshots of the Streamlit application, API documentation, exploratory data analysis, and deployed application will be added here as the deployment and documentation stages are completed.

Future Improvements
Potential future improvements include:

Comparing additional machine learning algorithms
Hyperparameter optimization
Cross-validation
More detailed model comparison
Improved handling of wine-quality classes
Explainable AI and advanced feature-importance analysis
Improved frontend design
Cloud deployment
Automated model retraining
Continuous integration and deployment

## Screenshots

Screenshots demonstrating the application interface, prediction workflow, and deployed application will be added after the deployment stage.

Author
Name: Segun Daramola

Cohort: 3MTT/DSN/WesOnline Mentorship Cohort 3

Programme: Data Science / AI / Machine Learning

Mentor / Training Credit
GworldSoft Solutions Limited / 3MTT (DSN/DeepTech_Ready/WesOnline)

Project Status
Current status: Machine Learning model developed, locally tested, FastAPI backend and Streamlit frontend integrated, and project repository published to GitHub.

Next milestone: Cloud deployment and connection of the frontend to the deployed backend.