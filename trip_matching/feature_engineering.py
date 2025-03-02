"""
Feature engineering module for the Trip Matching System.

This module handles the creation of features for machine learning models
by transforming raw truck and shipment data.
"""

import numpy as np
import pandas as pd


def create_features(truck_df, shipment_df):
    """
    Create features for all possible truck-shipment combinations

    Args:
        truck_df: DataFrame containing truck data
        shipment_df: DataFrame containing shipment data

    Returns:
        DataFrame containing features for all truck-shipment combinations
    """
    features = []

    for _, truck in truck_df.iterrows():
        for _, shipment in shipment_df.iterrows():
            # Calculate distance from truck to pickup
            dist_lat = shipment["pickup_lat"] - truck["current_lat"]
            dist_lon = shipment["pickup_lon"] - truck["current_lon"]
            distance_to_pickup = round(np.sqrt(dist_lat**2 + dist_lon**2) * 111, 1)

            # Calculate capacity utilization
            capacity_utilization = shipment["weight_tons"] / truck["capacity_tons"]
            # Create feature dictionary with simplified feature set
            feature = {
                "truck_id": truck["truck_id"],
                "shipment_id": shipment["shipment_id"],
                "distance_to_pickup_km": distance_to_pickup,
                "capacity_utilization": capacity_utilization,
                "driver_hours_sufficient": truck["driver_hours_available"]
                >= shipment["estimated_trip_hours"],
                "estimated_trip_hours": shipment["estimated_trip_hours"],
                "estimated_distance_km": shipment["estimated_distance_km"],
                "priority_high": 1 if shipment["priority"] == "High" else 0,
                "priority_medium": 1 if shipment["priority"] == "Medium" else 0,
                "maintenance_status_score": truck["maintenance_status_score"],
            }
            features.append(feature)

    return pd.DataFrame(features)


def select_features_for_model(historical_df):
    """
    Select and prepare features from historical data for model training

    Args:
        historical_df: DataFrame containing historical match data

    Returns:
        X: Feature matrix
        y: Target vector
    """
    features = historical_df.drop(columns=["match_score"])
    target = historical_df["match_score"]

    return features, target
