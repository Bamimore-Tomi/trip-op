"""
Model training and prediction module for the TruQ Trip Matching System.

This module handles the training of machine learning models and making predictions
for truck-shipment matches.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
    print(f"\nModel Training Results:")
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
