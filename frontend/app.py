#
...
# IMPORT REQUIRED LIBRARIES
#
...

# Import StreamLit for building the web application.

import streamlit as st

# Import Requests for communicating with the FastAPI backend.

import requests

#
...
# CONFIGURE STREAMLIT PAGE
#
...

# Configure the appearance of the StreamLit application.

st.set_page_config(
    page_title="Wine Quality Prediction System",
    page_icon="🍷",
    layout="centered"
)

#
...
# APPLICATION TITLE
#
...

st.title("🍷 Wine Quality Prediction System")

st.write(
    """
    This application predicts the quality of wine based on its physicochemical
    properties using a trained Random Forest Machine Learning model.
    """
)

#
...
# CREATE THE USER INPUT FORM
#
...

st.header("Enter Wine Properties")

# Collect User Inputs.

fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, format="%.2f")

volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, format="%.2f")

citric_acid = st.number_input("Citric Acid", min_value=0.0, format="%.2f")

residual_sugar = st.number_input("ResidualSugar", min_value=0.0, format="%.2f")

chlorides = st.number_input("Chlorides", min_value=0.0, format="%.3f")

free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, format="%.2f")

total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, format="%.2f")

density = st.number_input("Density", min_value=0.0, format="%.5f")

pH = st.number_input("pH", min_value=0.0, format="%.2f")

sulphates = st.number_input("Sulphates", min_value=0.0, format="%.2f")

alcohol = st.number_input("Alcohol", min_value=0.0, format="%.2f")

wine_type = st.selectbox(
    "Wine Type",
    options=[0,1],
    format_func=lambda x: "Red Wine" if x == 0 else "White Wine"
)

#
...
# CREATE THE PREDICT BUTTON
#
...

# Create a predict button that starts the prediction process.

predict_button = st.button("Predict Wine Quality")

#
...
# PREPARE THE INPUT DATA
#
...

#
...

input_data = {
    "fixed_acidity":fixed_acidity,
    "volatile_acidity":volatile_acidity,
    "citric_acid":citric_acid,
    "residual_sugar":residual_sugar,
    "chlorides":chlorides,
    "free_sulfur_dioxide":free_sulfur_dioxide,
    "total_sulfur_dioxide":total_sulfur_dioxide,
    "density":density,
    "pH":pH,
    "sulphates":sulphates,
    "alcohol":alcohol,
    "wine_type":wine_type
}

#
...
# CONNECT STREAMLIT FRONTEND TO FASTAPI BACKEND
#
...

# If the predict button is clicked.

if predict_button:
    try:
        # Send the input data to the FastAPI backend.
        response = requests.post(
            "https://wine-quality-prediction-j49j.onrender.com/predict",
            json=input_data
        )
        # Check if the request was successful.
        if response.status_code == 200:
            prediction = response.json()

            st.success(
                f"Predicted Wine Quality: {prediction['Predicted Wine Quality']}"
            )
        else:
            st.error("Prediction failed. Please check the API.")
    except Exception as e: st.error(f"Unable to connect to the backend.\n\n{e}")