"""
Data generation module for the TruQ Trip Matching System.

This module handles the creation of synthetic data for demonstration
purposes, including trucks, shipments, and historical matches.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_locations():
    """Generate base locations (distribution centers and common areas)"""
    return [
        {"name": "Lagos Center", "lat": 6.5244, "lon": 3.3792},
        {"name": "Ikeja Hub", "lat": 6.6018, "lon": 3.3515},
        {"name": "Lekki Point", "lat": 6.4698, "lon": 3.5852},
        {"name": "Abuja North", "lat": 9.0765, "lon": 7.3986},
        {"name": "Abuja Central", "lat": 9.0579, "lon": 7.4951},
        {"name": "Ibadan Main", "lat": 7.3775, "lon": 3.9470},
    ]


def generate_trucks(num_trucks=50):
    """
    Generate synthetic truck data

    Args:
        num_trucks: Number of trucks to generate

    Returns:
        DataFrame containing truck data
    """
    np.random.seed(42)
    random.seed(50)

    locations = generate_locations()

    # Generate truck data
    trucks = []
    for i in range(num_trucks):
        # Randomly select current location
        current_loc = random.choice(locations)

        # Generate truck data
        truck = {
            "truck_id": f"T{i+1:03d}",
            "capacity_tons": round(random.uniform(5, 15), 1),
            "current_lat": current_loc["lat"] + random.uniform(-0.05, 0.05),
            "current_lon": current_loc["lon"] + random.uniform(-0.05, 0.05),
            "driver_hours_available": round(random.uniform(2, 8), 1),
            "maintenance_status_score": random.choice([1, 2, 3]),
            "fuel_level_percent": random.randint(30, 100),
            "last_delivery_time": datetime.now()
            - timedelta(hours=random.randint(1, 48)),
        }
        trucks.append(truck)

    return pd.DataFrame(trucks)


def generate_shipments(num_shipments=100, days=7):
    """
    Generate synthetic shipment data

    Args:
        num_shipments: Number of shipments to generate
        days: Number of days into the future for delivery deadlines

    Returns:
        DataFrame containing shipment data
    """
    np.random.seed(48)  # Different seed for variation
    random.seed(70)

    locations = generate_locations()

    # Generate shipment data
    shipments = []
    for i in range(num_shipments):
        # Randomly select pickup and delivery locations
        pickup_loc = random.choice(locations)
        delivery_loc = random.choice([loc for loc in locations if loc != pickup_loc])

        # Calculate estimated distance and duration
        dist_lat = delivery_loc["lat"] - pickup_loc["lat"]
        dist_lon = delivery_loc["lon"] - pickup_loc["lon"]
        estimated_distance_km = round(
            np.sqrt(dist_lat**2 + dist_lon**2) * 111, 1
        )  # Rough conversion to km
        estimated_trip_hours = round(
            estimated_distance_km / 110 * (1 + random.uniform(-0.2, 0.4)), 1
        )  # Assuming 50 km/h average

        # Generate random delivery window
        hours_from_now = random.randint(2, days * 24)
        delivery_deadline = datetime.now() + timedelta(hours=hours_from_now)

        shipment = {
            "shipment_id": f"S{i+1:03d}",
            "weight_tons": round(random.uniform(0.5, 12), 1),
            "pickup_lat": pickup_loc["lat"],
            "pickup_lon": pickup_loc["lon"],
            "delivery_lat": delivery_loc["lat"],
            "delivery_lon": delivery_loc["lon"],
            "estimated_distance_km": estimated_distance_km,
            "estimated_trip_hours": estimated_trip_hours,
            "delivery_deadline": delivery_deadline,
            "priority": random.choice(["High", "Medium", "Low"]),
            "customer_id": f"C{random.randint(1, 20):03d}",
        }
        shipments.append(shipment)

    return pd.DataFrame(shipments)


def naive_score(
    capacity_utilization,
    driver_hours_sufficient,
    distance_to_pickup_km,
    priority,
    maintenance_score,
):
    """
    Calculate a match score using a simplified weight-based approach that doesn't
    differentiate between long and short distances.

    Args:
        capacity_utilization: Ratio of shipment weight to truck capacity
        driver_hours_sufficient: Boolean indicating if driver has enough hours
        distance_to_pickup_km: Distance from truck to pickup location in km
        priority: Priority of the shipment ('High', 'Medium', 'Low')
        maintenance_score: Maintenance status score (1-3, where 1 is BEST)

    Returns:
        Match score from 0-100
    """
    # Define component scores (0-100 scale for each factor)

    # 1. Capacity Utilization Score
    # Reward utilization up to 100%, then penalize for exceeding capacity
    if capacity_utilization <= 1.0:
        # Linear increase up to 100%
        capacity_score = capacity_utilization * 100
    else:
        # Penalize by 8 points for each percentage point over 100%
        capacity_score = max(0, 100 - (capacity_utilization - 1.0) * 800)

    # 2. Driver Hours Score (binary)
    driver_hours_score = 100 if driver_hours_sufficient else 0

    # 3. Distance Score - Stronger exponential decay for more differentiation
    # Using exponential decay to more strongly penalize longer distances
    distance_score = 100 * (0.95**distance_to_pickup_km)

    # For high-priority shipments, make distance even more important
    if priority == "High":
        # Apply a more aggressive scoring for high priority shipments
        distance_score = 100 * (0.98**distance_to_pickup_km)
        # This ensures that shorter distances have a much stronger advantage

    # 4. Maintenance Score - INVERTED (1 is BEST, 3 is WORST)
    # Convert from 1-3 scale (where 1 is best) to 0-100 scale
    maintenance_score_normalized = (4 - maintenance_score) / 3 * 100

    # Define fixed weights for all distances
    weights = {
        "capacity": 0.25,
        "distance": 0.40,  # Distance is now the dominant factor
        "maintenance": 0.20,
        "driver_hours": 0.15,
    }

    # Calculate weighted score
    final_score = (
        capacity_score * weights["capacity"]
        + distance_score * weights["distance"]
        + maintenance_score_normalized * weights["maintenance"]
        + driver_hours_score * weights["driver_hours"]
    )

    return final_score


def generate_historical_matches(num_matches=500):
    """
    Generate synthetic historical match data with outcomes for training

    Args:
        num_matches: Number of historical matches to generate

    Returns:
        DataFrame containing historical match data with outcomes
    """
    np.random.seed(44)  # Different seed for variation
    random.seed(30)

    # Generate historical matching data with outcomes for training
    trucks_df = generate_trucks(num_matches)
    shipments_df = generate_shipments(num_matches, days=3)
    trucks_shipments_df = trucks_df.join(shipments_df, how="outer")
    trucks_shipments_df["capacity_utilization"] = trucks_shipments_df.apply(
        lambda row: row["weight_tons"] * row["capacity_tons"], axis=1
    )
    trucks_shipments_df["driver_hours_sufficient"] = trucks_shipments_df.apply(
        lambda row: (row["driver_hours_available"] >= row["estimated_trip_hours"]),
        axis=1,
    )
    trucks_shipments_df["distance_to_pickup_km"] = trucks_shipments_df.apply(
        lambda row: round(
            np.sqrt(
                (row["pickup_lat"] - row["current_lat"]) ** 2
                + (row["pickup_lon"] - row["current_lon"]) ** 2
            )
            * 111,
            1,
        ),
        axis=1,
    )
    trucks_shipments_df["match_score"] = trucks_shipments_df.apply(
        lambda row: naive_score(
            row["capacity_utilization"],
            row["driver_hours_sufficient"],
            row["distance_to_pickup_km"],
            row["priority"],
            row["maintenance_status_score"],
        ),
        axis=1,
    )
    trucks_shipments_df["priority_high"] = trucks_shipments_df.apply(
        lambda row: 1 if row["priority"] == "High" else 0,
        axis=1,
    )
    trucks_shipments_df["priority_medium"] = trucks_shipments_df.apply(
        lambda row: 1 if row["priority"] == "Medium" else 0,
        axis=1,
    )

    historical_data = pd.DataFrame().assign(
        distance_to_pickup_km=trucks_shipments_df["distance_to_pickup_km"],
        capacity_utilization=trucks_shipments_df["capacity_utilization"],
        driver_hours_sufficient=trucks_shipments_df["driver_hours_sufficient"],
        estimated_trip_hours=trucks_shipments_df["estimated_trip_hours"],
        estimated_distance_km=trucks_shipments_df["estimated_distance_km"],
        priority_high=trucks_shipments_df["priority_high"],
        priority_medium=trucks_shipments_df["priority_medium"],
        maintenance_status_score=trucks_shipments_df["maintenance_status_score"],
        match_score=trucks_shipments_df["match_score"],
    )

    return historical_data


def generate_prediction_data(num_trucks=10, num_shipments=1):
    """
    Generate new data for prediction (not used in training)

    Args:
        num_trucks: Number of trucks to generate
        num_shipments: Number of shipments to generate

    Returns:
        Tuple of DataFrames (new_trucks_df, new_shipments_df)
    """
    # Use different random seeds for new data
    np.random.seed(99)
    random.seed(99)

    # Generate new trucks and shipments
    new_trucks_df = generate_trucks(num_trucks)
    new_shipments_df = generate_shipments(num_shipments, days=3)

    # Rename IDs to make it clear these are new
    new_trucks_df["truck_id"] = [f"NT{i+1:03d}" for i in range(len(new_trucks_df))]
    new_shipments_df["shipment_id"] = [
        f"NS{i+1:03d}" for i in range(len(new_shipments_df))
    ]

    return new_trucks_df, new_shipments_df
