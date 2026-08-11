#
...
# IMPORT REQUIRED LIBRARIES
#
...

# Import all the libraries required to to build the FastAPI backend
# application and enable real-time wine quality prediction.

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

#
...
# CREATE THE FASTAPI APPLICATION
#
...

# Create and configure the FastAPI application that will serve as
# the backend for the Wine Quality Prediction System.

app = FastAPI(
    title="Wine Quality Prediction API",
    description="Backend API for predicting wine quality using a trained Random Forest Classifier.",
    version="1.0"
)

#
...
# LOAD THE SAVED TRAINED MODEL AND SCALER
#
...

# Load the trained Random Forest model and the fitted StandardScaler
# into memory.

model = joblib.load("../models/random_forest_model.pkl")

scaler = joblib.load("../models/scaler.pkl")

#
...
# DEFINE INPUT DATA MODEL
#
...

# Define the structure and data types of the wine properties that users
# most provide when requesting a prediction.

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    wine_type: int

#
...
# CRATE HOME ENDPOINT
#
...

# Create a simple home endpoint that confirms the API is running
# successfully.

# Display a welcome message when the API root URL is accessed.

@app.get("/")
def home():
    return {
        "message": "Wine Quality Prediction API is Running Successfully!"
    }

#
...
# CREATE PREDICTION ENDPOINT
#
...

# Receive wine chemical properties from users, preprocess the data
# using the saved StandardScaler, apply the trained Random Forest
# model, and return the predicted wine quality.

@app.post("/predict")
def predict(data: WineInput):
    input_data = np.array([[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol,
        data.wine_type
    ]])
    # Standardize the the input features using the saved scaler.
    
    scaled_data = scaler.transform(input_data)

    # Generate the wine quality prediction.
    
    prediction = model.predict(scaled_data)

    # Return the predicted wine quality as a JSON response.
    return {
        "Predicted Wine Quality": int(prediction[0])
    }