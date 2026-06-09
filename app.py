import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points on Earth in kilometers."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def euclidean_distance(lat1, lon1, lat2, lon2):
    """Calculates an approximate Euclidean distance between two points in kilometers."""
    lat_diff = (lat2 - lat1) * 111
    lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))
    return np.sqrt(lat_diff**2 + lon_diff**2)


def manhattan_distance(lat1, lon1, lat2, lon2):
    """Calculates an approximate Manhattan distance between two points in kilometers."""
    lat_diff = np.abs(lat2 - lat1) * 111
    lon_diff = np.abs(lon2 - lon1) * 111 * np.cos(np.radians(lat1))
    return lat_diff + lon_diff


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing between two points in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))
    bearing_rad = np.arctan2(x, y)
    bearing_deg = np.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


def engineer_features(df):
    """Replicates the feature engineering steps from the Colab notebook."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour

    df["haversine_distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["euclidean_distance"] = euclidean_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["manhattan_distance"] = manhattan_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["bearing"] = calculate_bearing(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    columns_to_drop = [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    if "key" in df.columns:
        columns_to_drop.append("key")

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    expected_columns = [
        "year",
        "month",
        "day",
        "weekday",
        "hour",
        "haversine_distance",
        "euclidean_distance",
        "manhattan_distance",
        "bearing",
    ]
    reordered_columns = [col for col in expected_columns if col in df.columns] + [col for col in df.columns if col not in expected_columns]
    return df[reordered_columns]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if data is None:
            raise ValueError("Request body must be valid JSON.")

        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            raise ValueError("JSON body must be an object or array of objects.")

        processed_df = engineer_features(input_df)
        prediction = model.predict(processed_df)
        result = prediction.tolist()
        if len(result) == 1:
            result = result[0]
        return jsonify({"fare_amount": result, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
