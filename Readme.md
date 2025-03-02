# TruQ Trip Matching System

An ML-based system for optimizing truck-shipment matching in middle-mile logistics.

## Business Problem

TruQ faced significant challenges in efficiently matching available trucks with pending shipments:

- Low truck utilization rate of only 60%
- Each truck completing just 2 trips per day (vs. target of 3)
- Inability to service 20% of customer requests due to inefficient asset allocation
- High operational costs and missed revenue opportunities (~$200K annually)

## Solution

This prototype demonstrates the core ML system I developed in collaboration with our Data Science team to solve TruQ's truck-shipment matching challenge.

The system uses a machine learning model to predict optimal matches between available trucks and pending shipments based on multiple features:

- Truck location and capacity
- Shipment weight and dimensions
- Driver availability
- Distance calculations
- Delivery deadlines
- Historical performance

## Installation

```bash
poetry install
poetry run trip-matcher
```

## Project Structure

```
tr-presentation/
├── pyproject.toml
├── README.md
└── trip_matching/
    └── main.py
```

## Business Impact

- Increased truck utilization from 60% to 80%
- Improved daily trips per truck from 2 to 3
- Reduced unserviced customer requests from 20% to 8%
- Generated approximately $15K in additional monthly revenue
- Decreased dispatcher workload by 40%

## Demo

The main.py file demonstrates:

1. Generation of synthetic data representing trucks and shipments
2. Feature engineering for potential matches
3. Training of the matching model using historical data
4. Prediction of optimal matches using the trained model
5. Selection and presentation of the best matches

This prototype serves as a simplified version of the production system deployed at TruQ.
