import streamlit as st
import requests


def reset_form():
    """Clear all wine-property input fields."""
    for key in [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "wine_type",
    ]:
        st.session_state[key] = None


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="🍷 Wine Quality Prediction",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# Application introduction
# ---------------------------------------------------------

st.title("🍷 Wine Quality Prediction")

st.write(
    "Enter the 12 features used by the prediction model "
    "to obtain a predicted wine quality score."
)


# ---------------------------------------------------------
# Wine properties
# ---------------------------------------------------------

st.subheader("Wine Properties")

st.caption(
    "Enter the values for all 12 features used by the prediction model."
)


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Left column
# ---------------------------------------------------------

with col1:

    fixed_acidity = st.number_input(
        "Fixed Acidity (g/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="fixed_acidity"
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity (g/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="volatile_acidity"
    )

    citric_acid = st.number_input(
        "Citric Acid (g/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="citric_acid"
    )

    residual_sugar = st.number_input(
        "Residual Sugar (g/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="residual_sugar"
    )

    chlorides = st.number_input(
        "Chlorides (g/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.3f",
        key="chlorides"
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide (mg/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="free_sulfur_dioxide"
    )


# ---------------------------------------------------------
# Right column
# ---------------------------------------------------------

with col2:

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide (mg/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="total_sulfur_dioxide"
    )

    density = st.number_input(
        "Density (g/cm³)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.5f",
        key="density"
    )

    pH = st.number_input(
        "pH",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="pH"
    )

    sulphates = st.number_input(
        "Sulphates (g/L)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="sulphates"
    )

    alcohol = st.number_input(
        "Alcohol (% vol.)",
        min_value=0.0,
        value=None,
        placeholder="Enter value",
        format="%.2f",
        key="alcohol"
    )

    wine_type = st.selectbox(
        "Wine Type",
        options=[0, 1],
        index=None,
        placeholder="Select wine type",
        format_func=lambda x: (
            "Red Wine" if x == 0 else "White Wine"
        ),
        key="wine_type"
    )


# ---------------------------------------------------------
# Action buttons
# ---------------------------------------------------------

st.write("")

predict_button = st.button(
    "Predict Wine Quality",
    use_container_width=True,
    type="primary"
)

reset_button = st.button(
    "Reset",
    use_container_width=True,
    on_click=reset_form
)


# ---------------------------------------------------------
# Prepare input data
# ---------------------------------------------------------

input_data = {
    "fixed_acidity": fixed_acidity,
    "volatile_acidity": volatile_acidity,
    "citric_acid": citric_acid,
    "residual_sugar": residual_sugar,
    "chlorides": chlorides,
    "free_sulfur_dioxide": free_sulfur_dioxide,
    "total_sulfur_dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
    "wine_type": wine_type
}


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if predict_button:

    numeric_inputs = [
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]

    # Check for incomplete input
    if any(value is None for value in numeric_inputs) or wine_type is None:

        st.warning(
            "Please enter all wine properties before making a prediction."
        )

    else:

        try:

            response = requests.post(
                "https://wine-quality-prediction-backend.onrender.com/predict",
                json=input_data,
                timeout=60
            )

            if response.status_code == 200:

                prediction = response.json()

                predicted_quality = prediction[
                    "Predicted Wine Quality"
                ]

                st.divider()

                st.subheader("Prediction Result")

                st.metric(
                    label="Predicted Wine Quality",
                    value=predicted_quality
                )

            else:

                st.error(
                    "The prediction service returned an error."
                )

                with st.expander("Technical details"):
                    st.write(
                        f"Status Code: {response.status_code}"
                    )
                    st.code(response.text)

        except requests.exceptions.Timeout:

            st.error(
                "The prediction service took too long to respond. "
                "Please try again."
            )

        except requests.exceptions.RequestException:

            st.error(
                "Unable to connect to the prediction service. "
                "Please check that the backend is available and try again."
            )


# ---------------------------------------------------------
# About the application
# ---------------------------------------------------------

st.divider()

with st.expander("About this application"):

    st.write(
        "This application predicts wine quality using a trained "
        "Random Forest Classifier."
    )

    st.write(
        "The model uses 12 features representing the physicochemical "
        "properties and type of the wine sample."
    )

    st.write(
        "The Streamlit interface communicates with a FastAPI backend, "
        "which processes the input using the saved machine-learning "
        "model and scaler."
    )