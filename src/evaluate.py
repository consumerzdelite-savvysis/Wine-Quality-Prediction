import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import preprocess_wine_data
from utils import (
    RANDOM_FOREST_MODEL_PATH,
    REPORTS_DIR,
)


def evaluate_model():
    """
    Evaluate the trained Random Forest model using the test dataset.
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

    # Load the trained Random Forest model
    model = joblib.load(
        RANDOM_FOREST_MODEL_PATH
    )

    # Generate predictions
    y_pred = model.predict(
        X_test_scaled
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Generate confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    # Prepare evaluation results
    evaluation_results = {
        "model": "Random Forest Classifier",
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    # Make sure the reports directory exists
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save evaluation metrics
    metrics_path = REPORTS_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=4
        )

    print("Model evaluation completed successfully.")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy percentage: {accuracy * 100:.2f}%")
    print("Confusion matrix:")
    print(matrix)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    evaluate_model()