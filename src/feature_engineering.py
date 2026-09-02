import pandas as pd


# Features used by the Wine Quality Prediction model
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]


def select_model_features(wine_clean):
    """
    Select the predictor variables used by the machine learning models.

    The 'quality' column is excluded because it is the target variable.
    """

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in wine_clean.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = wine_clean[FEATURE_COLUMNS].copy()

    return X


def validate_feature_order(X):
    """
    Confirm that the model input contains the expected features
    in the correct order.
    """

    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Feature names or feature order do not match "
            "the expected model features."
        )

    return True


if __name__ == "__main__":
    from preprocess import (
        load_wine_data,
        clean_wine_data,
        encode_wine_type,
    )

    wine_df = load_wine_data()

    wine_clean = clean_wine_data(wine_df)

    wine_clean = encode_wine_type(wine_clean)

    X = select_model_features(wine_clean)

    validate_feature_order(X)

    print("Feature engineering completed successfully.")
    print("Number of model features:", X.shape[1])
    print("Feature order:")
    
    for number, feature in enumerate(X.columns, start=1):
        print(f"{number}. {feature}")