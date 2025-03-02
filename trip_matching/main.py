"""
Trip Matching System - Main Module

This module provides the entry point for the ML-based trip matching system.
"""

import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import local modules
from trip_matching.data_generator import (
    generate_historical_matches,
    generate_prediction_data,
)
from trip_matching.feature_engineering import create_features, select_features_for_model
from trip_matching.model import predict_matches, train_matching_model
from trip_matching.visualization import visualize_match_tables


def main(show_visualizations=False, save_visualizations=False):
    """
    Main function to demonstrate the trip matching system

    Args:
        show_visualizations: If True, display visualizations
        save_visualizations: If True, save visualizations to files
    """
    print("Trip Matching System")
    print("===================")

    # Generate training data
    print("Generating training data...")
    historical_df = generate_historical_matches(num_matches=500)
    X, y = select_features_for_model(historical_df)
    model, scaler, evaluation = train_matching_model(X, y)

    new_trucks_df, new_shipment_df = generate_prediction_data(
        num_trucks=5, num_shipments=2
    )

    # Create features for prediction data
    prediction_features_df = create_features(new_trucks_df, new_shipment_df)
    top_matches = predict_matches(
        model, scaler, prediction_features_df, new_trucks_df, new_shipment_df, top_n=5
    )
    grouped_by_shipment = (
        top_matches.groupby("shipment_id")
        .apply(lambda g: g.to_dict(orient="records"))
        .to_dict()
    )

    for shipment_id, trucks in grouped_by_shipment.items():
        print(f"\nAll Trucks Scored for Shipment {shipment_id}:")
        print("-" * 80)
        print(
            f"{'Rank':<5}{'Truck ID':<10}{'Score':<10}{'Distance':<12}{'Utilization':<15}{'Hours Sufficient'}"
        )
        print("-" * 80)

        for rank, row in enumerate(trucks, 1):
            print(
                f"{rank:<5}{row['truck_id']:<10}{row['match_score']:.1f}{'':<10}{row['distance_to_pickup_km']:.1f} km{'':<5}{row['capacity_utilization']:.1%}{'':<5}{'Yes' if row['driver_hours_sufficient'] else 'No'}"
            )

    # Show feature importance
    print("\nFeature Importance:")
    print("-" * 80)
    for i, (feature, importance) in enumerate(
        zip(
            evaluation["feature_importance"]["Feature"].values[:5],
            evaluation["feature_importance"]["Importance"].values[:5],
        )
    ):
        print(f"{i+1}. {feature}: {importance:.4f}")

    # Generate visualizations if requested
    if show_visualizations or save_visualizations:
        print("\nGenerating match table visualizations...")
        match_figures = visualize_match_tables(
            top_matches, new_trucks_df, new_shipment_df
        )

        if save_visualizations:
            # Save match table figures
            for i, fig in enumerate(match_figures):
                fig.savefig(
                    f"match_table_shipment_{i+1}.png", dpi=300, bbox_inches="tight"
                )

        if show_visualizations:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trip Matching System")
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Show visualizations"
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="Save visualizations to files"
    )
    args = parser.parse_args()

    main(show_visualizations=args.visualize, save_visualizations=args.save)
