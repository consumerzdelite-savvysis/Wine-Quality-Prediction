import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    RED_WINE_PATH,
    WHITE_WINE_PATH,
)


def load_wine_data():
    """
    Load and combine the red and white wine datasets.
    """

    red_wine = pd.read_csv(
        RED_WINE_PATH,
        sep=";"
    )

    white_wine = pd.read_csv(
        WHITE_WINE_PATH,
        sep=";"
    )

    red_wine["wine_type"] = "Red"
    white_wine["wine_type"] = "White"

    wine_df = pd.concat(
        [red_wine, white_wine],
        ignore_index=True
    )

    return wine_df


def clean_wine_data(wine_df):
    """
    Remove duplicate wine records from the combined dataset.
    """

    wine_clean = wine_df.drop_duplicates().copy()

    return wine_clean


def encode_wine_type(wine_clean):
    """
    Convert the categorical wine_type variable into numerical values.

    Red = 0
    White = 1
    """

    wine_clean = wine_clean.copy()

    wine_clean["wine_type"] = wine_clean["wine_type"].map({
        "Red": 0,
        "White": 1
    })

    return wine_clean


def prepare_features_and_target(wine_clean):
    """
    Separate predictor variables from the target variable.
    """

    X = wine_clean.drop("quality", axis=1)
    y = wine_clean["quality"]

    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Split the dataset into training and testing sets using the
    same 80/20 methodology established in the model-development
    notebook.

    The split is intentionally kept without stratification so that
    the reusable src pipeline reproduces the project's documented
    Random Forest evaluation and model-selection results.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize training and testing features.

    The scaler is fitted only on the training data to prevent
    information from the testing data from influencing the model.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_wine_data():
    """
    Execute the complete wine-data preprocessing pipeline.
    """

    wine_df = load_wine_data()

    wine_clean = clean_wine_data(wine_df)

    wine_clean = encode_wine_type(wine_clean)

    X, y = prepare_features_and_target(wine_clean)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    return (
        wine_df,
        wine_clean,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    )


if __name__ == "__main__":
    (
        wine_df,
        wine_clean,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = preprocess_wine_data()

    print("Original dataset shape:", wine_df.shape)
    print("Cleaned dataset shape:", wine_clean.shape)
    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Scaled training features shape:", X_train_scaled.shape)
    print("Scaled testing features shape:", X_test_scaled.shape)