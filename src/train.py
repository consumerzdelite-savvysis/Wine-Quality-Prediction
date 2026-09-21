"""
Train the Random Forest model for Wine Quality Prediction.

This script:
1. Loads and preprocesses the wine dataset.
2. Creates the training and testing datasets.
3. Validates the expected 12 model features and their order.
4. Uses the StandardScaler created during preprocessing.
5. Trains the Random Forest classifier.
6. Saves the trained model and scaler for later prediction/deployment.

The preprocessing function already fits the StandardScaler on the training
data and uses it to transform both the training and testing data. Therefore,
this script intentionally does NOT create another scaler.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess_wine_data
from feature_engineering import validate_feature_order
import utils


# ============================================================
# STEP 1: LOAD AND PREPROCESS THE DATA
# ============================================================
#
# preprocess_wine_data() performs the main data-preparation tasks:
#
# - Loads the red and white wine datasets
# - Combines the datasets
# - Removes duplicate records
# - Creates the wine_type feature
# - Separates the target variable (quality)
# - Splits the data into training and testing sets
# - Fits StandardScaler ONLY on the training data
# - Uses that scaler to transform both training and testing data
#
# The returned scaler is the same scaler that must be saved and later
# used by the prediction/deployment pipeline.

(
    wine_df,
    wine_clean,
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    scaler,
) = preprocess_wine_data()


# ============================================================
# STEP 2: VALIDATE THE MODEL FEATURES
# ============================================================
#
# The project uses exactly 12 predictor variables:
#
# 1. fixed acidity
# 2. volatile acidity
# 3. citric acid
# 4. residual sugar
# 5. chlorides
# 6. free sulfur dioxide
# 7. total sulfur dioxide
# 8. density
# 9. pH
# 10. sulphates
# 11. alcohol
# 12. wine_type
#
# validate_feature_order() confirms that the training and testing
# datasets contain these features in exactly the expected order.
#
# This is important because the Random Forest model must receive
# the same feature order during training and prediction.

validate_feature_order(X_train)
validate_feature_order(X_test)


# ============================================================
# STEP 3: CREATE THE RANDOM FOREST CLASSIFIER
# ============================================================
#
# The baseline Random Forest configuration used for this project is:
#
# - 100 decision trees
# - random_state=42 for reproducibility
#
# This is the documented model configuration that produced the
# project's 57.71% test accuracy.

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================
#
# The model is trained using the already-scaled training data.
#
# X_train_scaled contains the 12 validated predictor variables.
# y_train contains the corresponding wine quality scores.

random_forest.fit(X_train_scaled, y_train)


# ============================================================
# STEP 5: SAVE THE TRAINED MODEL
# ============================================================
#
# The trained Random Forest model is saved so that it can be loaded
# later by the prediction script and FastAPI backend.

joblib.dump(random_forest, utils.RANDOM_FOREST_MODEL_PATH)

print("Random Forest model trained successfully.")
print(f"Model saved to: {utils.RANDOM_FOREST_MODEL_PATH}")


# ============================================================
# STEP 6: SAVE THE SCALER
# ============================================================
#
# The scaler used during training must also be saved.
#
# During deployment, new user input must pass through this EXACT
# scaler before being sent to the Random Forest model.
#
# Saving the same scaler used during training prevents a mismatch
# between model training and future predictions.

joblib.dump(scaler, utils.SCALER_PATH)

print(f"Scaler saved to: {utils.SCALER_PATH}")


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\nTraining Summary")
print("----------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
print("Model: Random Forest Classifier")
print("Number of trees: 100")
print("Random state: 42")
print("Scaler: StandardScaler")