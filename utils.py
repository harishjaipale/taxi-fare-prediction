import pandas as pd
import numpy as np


def extract_time_features(df):
    """Extracts time-based features from 'pickup_datetime'."""
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['hour'] = df['pickup_datetime'].dt.hour
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points in kilometers."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def euclidean_distance(lat1, lon1, lat2, lon2):
    """Calculates an approximated Euclidean distance between two points (in km)."""
    lat_diff = (lat2 - lat1) * 111
    lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))
    return np.sqrt(lat_diff**2 + lon_diff**2)


def manhattan_distance(lat1, lon1, lat2, lon2):
    """Calculates an approximated Manhattan distance between two points (in km)."""
    lat_diff = np.abs(lat2 - lat1) * 111
    lon_diff = np.abs(lon2 - lon1) * 111 * np.cos(np.radians(lat1))
    return lat_diff + lon_diff


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing (direction) between two geographical points in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))
    bearing_rad = np.arctan2(x, y)
    bearing_deg = np.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


def apply_feature_engineering(df):
    """Applies feature engineering to a DataFrame with taxi trip data."""
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df = extract_time_features(df)
    df['haversine_distance'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['euclidean_distance'] = euclidean_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['manhattan_distance'] = manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['bearing'] = calculate_bearing(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    columns_to_drop = [
        'pickup_datetime',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
    ]
    if 'key' in df.columns:
        columns_to_drop.append('key')
    return df.drop(columns=[c for c in columns_to_drop if c in df.columns])
