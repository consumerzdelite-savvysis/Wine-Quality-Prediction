import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess_wine_data
from feature_engineering import (
    select_model_features,
    validate_feature_order,
)
from utils import (
    RANDOM_FOREST_MODEL_PATH,
    SCALER_PATH,
)


def train_random_forest():
    """
    Train the final Random Forest model using the project's
    established preprocessing and feature-engineering pipeline.
    """

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

    # Select the exact features used by the model
    X_train_features = select_model_features(
        X_train
    )

    X_test_features = select_model_features(
        X_test
    )

    # Confirm feature names and order
    validate_feature_order(X_train_features)
    validate_feature_order(X_test_features)

    # Train the final Random Forest model
    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    random_forest.fit(
        X_train_scaled,
        y_train
    )

    # Save the trained model
    joblib.dump(
        random_forest,
        RANDOM_FOREST_MODEL_PATH
    )

    # Save the scaler used during preprocessing
    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("Random Forest training completed successfully.")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("Number of features:", X_train_features.shape[1])
    print("Model: Random Forest Classifier")
    print("Number of trees:", random_forest.n_estimators)
    print("Random state:", random_forest.random_state)
    print("Model saved to:", RANDOM_FOREST_MODEL_PATH)
    print("Scaler saved to:", SCALER_PATH)


if __name__ == "__main__":
    train_random_forest()