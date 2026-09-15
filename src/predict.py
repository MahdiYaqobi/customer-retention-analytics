"""
Customer Churn Prediction - Inference

This script accepts raw IBM Telco Customer Churn-style customer data,
applies the required data cleaning and feature engineering steps,
loads the trained Logistic Regression pipeline,
and predicts churn probability.

Business threshold:
    0.30

A customer is classified as "Churn" when the predicted probability
is greater than or equal to the selected threshold.
"""

from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

THRESHOLD = 0.30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model_logreg.pkl"


# ============================================================
# 1. Data Cleaning
# ============================================================

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw customer data using the same logic as Notebook 03.

    Expected raw IBM Telco column names include:
        customerID
        SeniorCitizen
        PhoneService
        InternetService
        TotalCharges
        ...

    Returns:
        Cleaned DataFrame with standardized column names
        and numeric total_charges.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------

    df.columns = df.columns.str.lower().str.strip()

    df.rename(
        columns={
            "customerid": "customer_id",
            "seniorcitizen": "senior_citizen",
            "phoneservice": "phone_service",
            "multiplelines": "multiple_lines",
            "internetservice": "internet_service",
            "onlinesecurity": "online_security",
            "onlinebackup": "online_backup",
            "deviceprotection": "device_protection",
            "techsupport": "tech_support",
            "streamingtv": "streaming_tv",
            "streamingmovies": "streaming_movies",
            "paperlessbilling": "paperless_billing",
            "paymentmethod": "payment_method",
            "monthlycharges": "monthly_charges",
            "totalcharges": "total_charges",
        },
        inplace=True,
    )

    # --------------------------------------------------------
    # Handle TotalCharges
    # --------------------------------------------------------

    if "total_charges" in df.columns:

        # Raw IBM data contains blank strings.
        df["total_charges"] = (
            df["total_charges"]
            .replace(" ", np.nan)
            .astype(float)
        )

        # For new customers with zero tenure,
        # missing total charges represent zero accumulated charges.
        df["total_charges"] = df["total_charges"].fillna(0)

    return df


# ============================================================
# 2. Feature Engineering
# ============================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the engineered features used during model training.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Normalize service categories
    # --------------------------------------------------------

    service_columns = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    for column in service_columns:
        if column in df.columns:
            df[column] = df[column].replace(
                "No internet service",
                "No",
            )

    # --------------------------------------------------------
    # Total number of subscribed services
    # --------------------------------------------------------

    service_columns = [
        "phone_service",
        "internet_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    service_mapping = {
        "Yes": 1,
        "No": 0,
        "DSL": 1,
        "Fiber optic": 1,
        "No internet service": 0,
        "No phone service": 0,
    }

    df["total_services"] = (
        df[service_columns]
        .replace(service_mapping)
        .sum(axis=1)
        .astype(int)
    )

    # --------------------------------------------------------
    # New customer flag
    # --------------------------------------------------------

    df["is_new_customer"] = (
        df["tenure"] <= 12
    ).astype(int)

    # --------------------------------------------------------
    # Tenure groups
    # --------------------------------------------------------

    bins = [0, 12, 24, 48, 72]
    labels = [
        "New",
        "Developing",
        "Established",
        "Loyal",
    ]

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    # --------------------------------------------------------
    # Streaming services
    # --------------------------------------------------------

    streaming_columns = [
        "streaming_tv",
        "streaming_movies",
    ]

    df["streaming_services"] = (
        df[streaming_columns]
        .replace({
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
        })
        .sum(axis=1)
        .astype(int)
    )

    # --------------------------------------------------------
    # Security services
    # --------------------------------------------------------

    security_columns = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
    ]

    df["has_security_services"] = (
        df[security_columns]
        .replace({
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
        })
        .sum(axis=1)
        > 0
    ).astype(int)

    return df


# ============================================================
# 3. Prepare Customer Data
# ============================================================

def prepare_customer_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw customer data for the trained ML pipeline.

    Steps:
        Raw data
            ↓
        Cleaning
            ↓
        Feature engineering
            ↓
        Remove identifier / target
            ↓
        Model-ready data
    """

    df = clean_raw_data(df)

    df = create_features(df)

    # --------------------------------------------------------
    # Remove columns that should not enter the model
    # --------------------------------------------------------

    columns_to_drop = []

    # Customer ID is only an identifier.
    if "customer_id" in df.columns:
        columns_to_drop.append("customer_id")

    # Churn is the target variable and should not be used
    # when predicting new customers.
    if "churn" in df.columns:
        columns_to_drop.append("churn")

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    return df


# ============================================================
# 4. Validate Input
# ============================================================

def validate_input(df: pd.DataFrame) -> None:
    """
    Validate that all features required by the trained model
    are present in the input data.
    """

    required_columns = [
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "tenure",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


# ============================================================
# 5. Predict Churn
# ============================================================

def predict_churn(
    customer_data: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """
    Predict churn probability for one or more customers.

    Parameters
    ----------
    customer_data:
        Raw or cleaned customer DataFrame.

    model_path:
        Path to the trained Logistic Regression pipeline.

    threshold:
        Classification threshold used to convert probability
        into a Churn / Stay prediction.

    Returns
    -------
    pd.DataFrame
        Original customer data plus:
            churn_probability
            churn_prediction
            churn_label
    """

    # Load trained pipeline.
    model = joblib.load(model_path)

    # Clean raw data.
    cleaned_data = clean_raw_data(customer_data)

    # Validate required columns before feature engineering.
    validate_input(cleaned_data)

    # Create model features.
    prepared_data = create_features(cleaned_data)

    # Remove identifier and target from model input.
    model_input = prepared_data.drop(
        columns=["customer_id", "churn"],
        errors="ignore",
    )

    # Predict probability of churn.
    churn_probability = model.predict_proba(
        model_input
    )[:, 1]

    # Apply business-selected threshold.
    churn_prediction = (
        churn_probability >= threshold
    ).astype(int)

    # Keep original customer data in the output.
    results = customer_data.copy()

    results["churn_probability"] = churn_probability

    results["churn_prediction"] = churn_prediction

    results["churn_label"] = (
        results["churn_prediction"]
        .map({
            0: "Stay",
            1: "Churn",
        })
    )

    return results


# ============================================================
# 6. Main (CLI Entry Point)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict customer churn from a CSV file."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path to save predictions CSV."
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # Load raw customer data
    customers = pd.read_csv(input_path)

    # Predict churn
    predictions = predict_churn(customers)

    # Create default output path if not provided
    if args.output is None:
        output_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "churn_predictions.csv"
        )
    else:
        output_path = Path(args.output)

    # Create output directory if necessary
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions.to_csv(output_path, index=False)

    print("\nCustomer Churn Prediction")
    print("-" * 30)

    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Customers:   {len(predictions)}")

    print("\nPrediction summary:")
    print(predictions["churn_label"].value_counts())

    print("\nFirst predictions:")
    print(
        predictions[
            [
                "customerID",
                "churn_probability",
                "churn_prediction",
                "churn_label",
            ]
        ].head(10).to_string(index=False)
    )