"""
Visualization module for the Trip Matching System.

This module provides a function to visualize match results using matplotlib tables.
"""

import matplotlib.pyplot as plt


def visualize_match_tables(top_matches, new_trucks_df, new_shipment_df):
    """
    Create table visualizations of truck-shipment matches

    Args:
        top_matches: DataFrame containing the top matches
        new_trucks_df: DataFrame containing truck data
        new_shipment_df: DataFrame containing shipment data

    Returns:
        List of matplotlib figure objects (one figure per shipment)
    """
    # Group matches by shipment
    grouped_by_shipment = top_matches.groupby("shipment_id")
    figures = []

    # Define colors for different values
    score_colors = {
        "high": "#99ff99",  # Light green for high scores
        "medium": "#ffff99",  # Light yellow for medium scores
        "low": "#ff9999",  # Light red for low scores
    }

    # For each shipment, create a separate figure with a table
    for shipment_id, group in grouped_by_shipment:
        # Get shipment details
        shipment = new_shipment_df[new_shipment_df["shipment_id"] == shipment_id].iloc[
            0
        ]

        # Sort by match_score descending
        matches = group.sort_values("match_score", ascending=False).reset_index(
            drop=True
        )

        # Prepare table data
        table_data = []
        for _, match in matches.iterrows():
            table_data.append(
                [
                    match["truck_id"],
                    f"{match['match_score']:.1f}",
                    f"{match['distance_to_pickup_km']:.1f} km",
                    f"{match['capacity_utilization']:.1%}",
                    "Yes" if match["driver_hours_sufficient"] else "No",
                    (
                        "High"
                        if match["priority_high"] == 1
                        else ("Medium" if match["priority_medium"] == 1 else "Low")
                    ),
                    f"{match['estimated_trip_hours']:.1f} hrs",
                    f"{match['estimated_distance_km']:.1f} km",
                    f"{match['maintenance_status_score']}",
                ]
            )

        # Create figure and axes for this shipment
        fig = plt.figure(figsize=(10, len(matches) * 0.5 + 2))
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Set title
        shipment_priority = (
            "High"
            if shipment.get("priority_high", False)
            else ("Medium" if shipment.get("priority_medium", False) else "Low")
        )
        if "priority" in shipment:
            shipment_priority = shipment["priority"]

        title = (
            f"Match Results for Shipment {shipment_id} - {shipment_priority} Priority"
        )
        ax.set_title(title, fontsize=14)

        # Column labels
        col_labels = [
            "Truck ID",
            "Score",
            "Distance\nto Pickup",
            "Capacity\nUtilization",
            "Hours\nSufficient",
            "Priority",
            "Trip\nHours",
            "Route\nDistance",
            "Maintenance\nScore",
        ]

        # Create the table
        table = ax.table(
            cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center"
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Make columns with more text wider
        table.auto_set_column_width([0, 2, 3, 4, 5, 6, 7, 8])

        # Style header
        for j in range(len(col_labels)):
            header_cell = table._cells[(0, j)]
            header_cell.set_text_props(weight="bold")
            header_cell.set_facecolor("#4472C4")
            header_cell.set_text_props(color="white")

        # Simple coloring for cells
        for i, row in enumerate(table_data):
            row_idx = i + 1  # Skip header row

            # Color score cell
            score_value = float(row[1])
            score_cell = table._cells[(row_idx, 1)]
            if score_value >= 70:
                score_cell.set_facecolor(score_colors["high"])
            elif score_value >= 50:
                score_cell.set_facecolor(score_colors["medium"])
            else:
                score_cell.set_facecolor(score_colors["low"])

            # Color distance cell
            distance_cell = table._cells[(row_idx, 2)]
            distance_value = float(row[2].split()[0])
            if distance_value <= 20:
                distance_cell.set_facecolor(score_colors["high"])
            elif distance_value <= 50:
                distance_cell.set_facecolor(score_colors["medium"])
            else:
                distance_cell.set_facecolor(score_colors["low"])

            # Color utilization cell
            util_cell = table._cells[(row_idx, 3)]
            util_value = float(row[3].strip("%")) / 100
            if 0.6 <= util_value <= 0.9:
                util_cell.set_facecolor(score_colors["high"])
            elif 0.3 <= util_value < 0.6 or 0.9 < util_value <= 1.0:
                util_cell.set_facecolor(score_colors["medium"])
            else:
                util_cell.set_facecolor(score_colors["low"])

            # Color hours sufficient cell
            hours_cell = table._cells[(row_idx, 4)]
            hours_cell.set_facecolor(
                score_colors["high"] if row[4] == "Yes" else score_colors["low"]
            )

            # Color priority cell
            priority_cell = table._cells[(row_idx, 5)]
            if row[5] == "High":
                priority_cell.set_facecolor("#ff9999")
            elif row[5] == "Medium":
                priority_cell.set_facecolor("#ffff99")
            else:
                priority_cell.set_facecolor("#99ff99")

        plt.tight_layout()
        figures.append(fig)

    return figures
