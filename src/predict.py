import joblib
import pandas as pd

from feature_engineering import FEATURE_COLUMNS
from utils import (
    RANDOM_FOREST_MODEL_PATH,
    SCALER_PATH,
)


def load_model_and_scaler():
    """
    Load the trained Random Forest model and fitted scaler.
    """

    model = joblib.load(
        RANDOM_FOREST_MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler


def prepare_prediction_input(input_data):
    """
    Prepare a single wine record for prediction.

    The input must contain the same 12 features used
    during model training.
    """

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )

    return input_df


def predict_wine_quality(input_data):
    """
    Predict the quality of a single wine sample.
    """

    model, scaler = load_model_and_scaler()

    input_df = prepare_prediction_input(
        input_data
    )

    input_scaled = scaler.transform(
        input_df
    )

    prediction = model.predict(
        input_scaled
    )

    return int(prediction[0])


if __name__ == "__main__":

    # Example wine sample
    sample_wine = {
        "fixed acidity": 7.0,
        "volatile acidity": 0.27,
        "citric acid": 0.36,
        "residual sugar": 20.7,
        "chlorides": 0.045,
        "free sulfur dioxide": 45.0,
        "total sulfur dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
        "wine_type": 1,
    }

    prediction = predict_wine_quality(
        sample_wine
    )

    print("Wine quality prediction completed successfully.")
    print("Predicted wine quality:", prediction)