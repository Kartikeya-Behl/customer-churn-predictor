import joblib
import pandas as pd


def load_artifacts():
    """Load the preprocessor, trained XGBoost model, and metadata threshold."""
    try:
        preprocessor = joblib.load("churn_preprocessor.joblib")
        model = joblib.load("churn_xgb_model.joblib")
        metadata = joblib.load("model_metadata.joblib")
        return preprocessor, model, metadata.get("optimal_threshold", 0.50)
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        raise


def predict_churn(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw customer features and generate churn predictions with probabilities."""
    preprocessor, model, threshold = load_artifacts()

    # Preprocess raw features
    X_prepared = preprocessor.transform(raw_df)

    # Get predicted probabilities for Churn (class 1)
    churn_probabilities = model.predict_proba(X_prepared)[:, 1]

    # Apply the tuned decision threshold
    predictions = (churn_probabilities >= threshold).astype(int)

    # Append results to a copy of the input dataframe
    results_df = raw_df.copy()
    results_df["Churn_Probability"] = churn_probabilities.round(4)
    results_df["Predicted_Churn"] = predictions

    return results_df


if __name__ == "__main__":
    # Load dataset for batch inference (replace with your test CSV or input data)
    data_path = "E-Commerce_Churn_Data.csv"

    try:
        raw_data = pd.read_csv(data_path)

        # Drop target column if present in the raw input file
        if "Churn" in raw_data.columns:
            raw_data = raw_data.drop(columns=["Churn"])

        # Run inference on the first 5 records as a sample
        sample_batch = raw_data.head(5)
        predictions_df = predict_churn(sample_batch)

        print("\n=== Inference Output ===")
        print(
            predictions_df[
                ["Tenure", "Complain", "Churn_Probability", "Predicted_Churn"]
            ]
        )

    except Exception as err:
        print(f"Failed to run inference pipeline: {err}")