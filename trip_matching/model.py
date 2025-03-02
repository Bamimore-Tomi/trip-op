"""
Model training and prediction module for the TruQ Trip Matching System.

This module handles the training of machine learning models and making predictions
for truck-shipment matches.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_matching_model(X, y, n_estimators=100, random_state=42, test_size=0.2):
    """
    Train a model to predict match scores

    Args:
        X: Feature matrix
        y: Target vector
        n_estimators: Number of trees in the random forest
        random_state: Random seed for reproducibility
        test_size: Proportion of data to use for testing

    Returns:
        model: Trained model
        scaler: Fitted scaler
        evaluation_metrics: Dictionary containing evaluation metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Get feature importance
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    # Print a summary of the model
    print("\nModel Training Results:")
    print(f"R² on training set: {train_score:.3f}")
    print(f"R² on test set: {test_score:.3f}")
    print(f"RMSE on test set: {rmse:.3f}")

    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    # Evaluation metrics
    evaluation_metrics = {
        "train_r2": train_score,
        "test_r2": test_score,
        "rmse": rmse,
        "feature_importance": feature_importance,
    }

    return model, scaler, evaluation_metrics


def predict_matches(model, scaler, features_df, trucks_df, shipments_df, top_n=3):
    """
    Predict match scores and return the best matches

    Args:
        model: Trained model
        scaler: Fitted scaler
        features_df: DataFrame containing features for all truck-shipment combinations
        trucks_df: DataFrame containing truck data
        shipments_df: DataFrame containing shipment data
        top_n: Number of top matches to return

    Returns:
        DataFrame containing the best matches
    """
    # Prepare features for prediction
    pred_features = features_df.drop(columns=["truck_id", "shipment_id"])

    # Scale features
    pred_features_scaled = scaler.transform(pred_features)

    # Predict scores
    predicted_scores = model.predict(pred_features_scaled)

    # Add predictions to the features dataframe
    features_df = features_df.copy()  # Create a copy to avoid modifying the original
    features_df["match_score"] = predicted_scores

    # Sort by predicted score
    sorted_matches = features_df.sort_values("match_score", ascending=False)

    return sorted_matches


def save_model(model, scaler, metrics, feature_names, model_dir="models"):
    """
    Save the trained model and associated artifacts

    Args:
        model: Trained model
        scaler: Fitted scaler
        metrics: Evaluation metrics
        feature_names: List of feature names
        model_dir: Directory to save model artifacts
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"trip_matching_model_{timestamp}")

    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Save model and scaler
    joblib.dump(model, os.path.join(model_path, "model.joblib"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.joblib"))

    # Save feature names
    pd.Series(feature_names).to_csv(
        os.path.join(model_path, "feature_names.csv"),
        index=False,
        header=["feature_name"],
    )

    # Save metrics
    metrics_df = pd.DataFrame(
        {
            "metric": ["train_r2", "test_r2", "rmse", "training_timestamp"],
            "value": [
                metrics["train_r2"],
                metrics["test_r2"],
                metrics["rmse"],
                metrics["training_timestamp"],
            ],
        }
    )
    metrics_df.to_csv(os.path.join(model_path, "metrics.csv"), index=False)

    # Save feature importance
    metrics["feature_importance"].to_csv(
        os.path.join(model_path, "feature_importance.csv"), index=False
    )

    print(f"\nModel artifacts saved to: {model_path}")
    return model_path


def load_model(model_dir):
    """
    Load a saved model and its artifacts

    Args:
        model_dir: Directory containing the model artifacts

    Returns:
        model: Loaded model
        scaler: Loaded scaler
        feature_names: List of feature names
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    feature_names = pd.read_csv(os.path.join(model_dir, "feature_names.csv"))[
        "feature_name"
    ].tolist()

    return model, scaler, feature_names
    return model, scaler, feature_names
