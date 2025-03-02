"""
Visualization module for the Trip Matching System.

This module provides functions to visualize match results.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def visualize_match_tables(top_matches, new_trucks_df, new_shipment_df):
    """
    Create and display nice-looking tables of truck matches for each shipment

    Args:
        top_matches: DataFrame containing the top matches
        new_trucks_df: DataFrame containing truck data
        new_shipment_df: DataFrame containing shipment data

    Returns:
        List of matplotlib figure objects (one for each shipment)
    """
    # Group matches by shipment
    grouped_by_shipment = top_matches.groupby("shipment_id")
    figures = []

    # Create a custom colormap for the scores
    score_cmap = LinearSegmentedColormap.from_list(
        "ScoreMap",
        [(0.3, "#ff9999"), (0.5, "#ffff99"), (0.7, "#99ff99"), (1, "#4CAF50")],
    )

    # Create a custom colormap for utilization
    util_cmap = LinearSegmentedColormap.from_list(
        "UtilizationMap", [(0, "#ffcc99"), (0.6, "#99ccff"), (0.9, "#3366ff")]
    )

    for shipment_id, group in grouped_by_shipment:
        # Get shipment details
        shipment = new_shipment_df[new_shipment_df["shipment_id"] == shipment_id].iloc[
            0
        ]

        # Sort by score descending
        matches = group.sort_values("score", ascending=False).reset_index(drop=True)

        # Prepare table data
        table_data = []
        for _, match in matches.iterrows():
            # Get truck details
            truck = new_trucks_df[new_trucks_df["truck_id"] == match["truck_id"]].iloc[
                0
            ]

            # Format match data for display
            table_data.append(
                [
                    match["truck_id"],
                    f"{match['score']:.1f}",
                    f"{match['distance_to_pickup_km']:.1f} km",
                    f"{match['capacity_utilization']:.1%}",
                    "✓" if match["driver_hours_sufficient"] else "✗",
                    f"{match['priority']}",
                ]
            )

        # Convert to numpy array for easier slicing
        table_data = np.array(table_data)

        # Set up the figure and axes
        fig = plt.figure(figsize=(12, len(matches) * 0.5 + 3), dpi=100)
        ax = fig.add_subplot(111)

        # Hide axes
        ax.axis("off")
        ax.axis("tight")

        # Set title with shipment details
        title = (
            f"Match Results for Shipment {shipment_id}\n"
            f"Weight: {shipment['weight_tons']:.1f} tons | "
            f"Distance: {shipment['estimated_distance_km']:.1f} km | "
            f"Priority: {shipment['priority']}"
        )

        ax.set_title(title, fontsize=14, pad=20)

        # Column labels
        col_labels = [
            "Truck ID",
            "Match Score",
            "Distance",
            "Capacity\nUtilization",
            "Hours\nSufficient",
            "Priority",
        ]

        # Create the table
        table = ax.table(
            cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center"
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)  # Adjust the height of rows

        # Make the header row bold with a different background
        for j, cell in enumerate(table._cells[(0, j)] for j in range(len(col_labels))):
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white")

        # Color cells based on values
        for i in range(len(table_data)):
            row_idx = i + 1  # +1 because of the header row

            # Score coloring
            score_cell = table._cells[(row_idx, 1)]
            score_value = float(table_data[i, 1])
            score_cell.set_facecolor(score_cmap(score_value / 100))

            # Distance coloring - lower is better
            distance_cell = table._cells[(row_idx, 2)]
            distance_value = float(table_data[i, 2].split()[0])
            # Normalize distance - assume 0-100km range
            norm_distance = max(0, min(1, 1 - (distance_value / 100)))
            distance_cell.set_facecolor(score_cmap(norm_distance))

            # Utilization coloring
            util_cell = table._cells[(row_idx, 3)]
            util_value = float(table_data[i, 3].strip("%")) / 100
            util_cell.set_facecolor(util_cmap(util_value))

            # Driver hours coloring
            hours_cell = table._cells[(row_idx, 4)]
            hours_cell.set_facecolor(
                "#99ff99" if table_data[i, 4] == "✓" else "#ff9999"
            )

            # Priority coloring
            priority_cell = table._cells[(row_idx, 5)]
            if table_data[i, 5] == "High":
                priority_cell.set_facecolor("#ff9999")
            elif table_data[i, 5] == "Medium":
                priority_cell.set_facecolor("#ffff99")
            else:
                priority_cell.set_facecolor("#99ff99")

        plt.tight_layout()
        figures.append(fig)

    return figures
