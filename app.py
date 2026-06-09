import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
API_KEY = os.environ.get("API_KEY", "my-secret-key")


@app.get("/")
def home():
    return render_template_string(
        """
        <!doctype html>
        <html lang=\"en\">
        <head>
          <meta charset=\"utf-8\" />
          <title>Taxi Fare Predictor</title>
          <style>
            :root {
              --bg: #07111f;
              --panel: rgba(10, 22, 38, 0.92);
              --panel-2: rgba(16, 30, 48, 0.95);
              --accent: #5eead4;
              --accent-2: #8b5cf6;
              --text: #eff6ff;
              --muted: #bfdbfe;
              --shadow: 0 18px 45px rgba(15, 23, 42, 0.45);
            }
            * { box-sizing: border-box; }
            body {
              margin: 0;
              min-height: 100vh;
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
              color: var(--text);
              background:
                radial-gradient(circle at top, rgba(94, 234, 212, 0.08), transparent 25%),
                linear-gradient(145deg, #020617 0%, #07111f 45%, #111827 100%);
            }
            .shell {
              width: min(1120px, calc(100vw - 2rem));
              margin: 0 auto;
              padding: 32px 0 48px;
            }
            .hero {
              display: grid;
              grid-template-columns: 1.1fr 0.9fr;
              gap: 24px;
              align-items: stretch;
            }
            .card {
              background: var(--panel);
              border: 1px solid rgba(148, 163, 184, 0.18);
              border-radius: 24px;
              box-shadow: var(--shadow);
              backdrop-filter: blur(12px);
              padding: 24px;
            }
            h1 {
              margin: 0 0 10px;
              font-size: clamp(2rem, 5vw, 3rem);
              line-height: 1.05;
            }
            .lead {
              color: var(--muted);
              font-size: 1.02rem;
              line-height: 1.5;
            }
            .badge {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              padding: 8px 12px;
              border-radius: 999px;
              background: linear-gradient(135deg, rgba(94, 234, 212, 0.15), rgba(139, 92, 246, 0.2));
              color: #d8f8f3;
              font-size: 0.92rem;
              border: 1px solid rgba(148, 163, 184, 0.18);
              margin-bottom: 16px;
            }
            form {
              display: grid;
              gap: 10px;
            }
            .grid-2 {
              display: grid;
              grid-template-columns: repeat(2, 1fr);
              gap: 10px;
            }
            label {
              font-size: 0.92rem;
              color: var(--muted);
            }
            input {
              width: 100%;
              border-radius: 14px;
              border: 1px solid rgba(148, 163, 184, 0.22);
              background: var(--panel-2);
              color: var(--text);
              padding: 12px 12px;
              font-size: 0.98rem;
              outline: none;
              transition: border-color 0.2s, transform 0.2s;
            }
            input:focus {
              border-color: var(--accent);
              transform: translateY(-1px);
            }
            button {
              margin-top: 6px;
              border: none;
              border-radius: 14px;
              padding: 12px 14px;
              color: #04111d;
              font-weight: 700;
              font-size: 1rem;
              cursor: pointer;
              background: linear-gradient(135deg, var(--accent), #a78bfa);
              box-shadow: 0 12px 24px rgba(94, 234, 212, 0.2);
              transition: transform 0.15s, box-shadow 0.2s;
            }
            button:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(139, 92, 246, 0.28); }
            #result {
              margin-top: 14px;
              min-height: 28px;
              color: #cffafe;
              font-weight: 700;
            }
            .mini-note {
              color: var(--muted);
              font-size: 0.92rem;
              line-height: 1.4;
            }
            @media (max-width: 920px) {
              .hero { grid-template-columns: 1fr; }
              .grid-2 { grid-template-columns: 1fr; }
            }
          </style>
        </head>
        <body>
          <div class=\"shell\">
            <section class=\"hero\">
              <article class=\"card\">
                <div class=\"badge\">✨ Live Taxi Fare Estimator</div>
                <h1>Predict fares in a polished, modern interface.</h1>
                <p class=\"lead\">Use the form below to send a real prediction request to the API and watch the result appear instantly.</p>
              </article>
              <article class=\"card\">
                <form id=\"fareForm\">
                  <label for=\"pickup_datetime\">Pickup date &amp; time</label>
                  <input id=\"pickup_datetime\" type=\"text\" value=\"2024-01-15 08:30:00\" placeholder=\"pickup_datetime\" required />
                  <div class=\"grid-2\">
                    <div>
                      <label for=\"pickup_latitude\">Pickup latitude</label>
                      <input id=\"pickup_latitude\" type=\"number\" step=\"any\" value=\"40.721\" placeholder=\"pickup_latitude\" required />
                    </div>
                    <div>
                      <label for=\"pickup_longitude\">Pickup longitude</label>
                      <input id=\"pickup_longitude\" type=\"number\" step=\"any\" value=\"-73.987\" placeholder=\"pickup_longitude\" required />
                    </div>
                  </div>
                  <div class=\"grid-2\">
                    <div>
                      <label for=\"dropoff_latitude\">Dropoff latitude</label>
                      <input id=\"dropoff_latitude\" type=\"number\" step=\"any\" value=\"40.751\" placeholder=\"dropoff_latitude\" required />
                    </div>
                    <div>
                      <label for=\"dropoff_longitude\">Dropoff longitude</label>
                      <input id=\"dropoff_longitude\" type=\"number\" step=\"any\" value=\"-73.97\" placeholder=\"dropoff_longitude\" required />
                    </div>
                  </div>
                  <label for=\"passenger_count\">Passenger count</label>
                  <input id=\"passenger_count\" type=\"number\" value=\"1\" placeholder=\"passenger_count\" required />
                  <button type=\"submit\">Predict Fare</button>
                  <p class=\"mini-note\">Tip: this page uses the same protected API endpoint and model pipeline you deployed.</p>
                </form>
                <div id=\"result\"></div>
              </article>
            </section>
          </div>
          <script>
            const form = document.getElementById('fareForm');
            const result = document.getElementById('result');
            form.addEventListener('submit', async (e) => {
              e.preventDefault();
              result.textContent = 'Predicting...';
              const payload = {
                pickup_datetime: document.getElementById('pickup_datetime').value,
                pickup_latitude: parseFloat(document.getElementById('pickup_latitude').value),
                pickup_longitude: parseFloat(document.getElementById('pickup_longitude').value),
                dropoff_latitude: parseFloat(document.getElementById('dropoff_latitude').value),
                dropoff_longitude: parseFloat(document.getElementById('dropoff_longitude').value),
                passenger_count: parseInt(document.getElementById('passenger_count').value, 10)
              };
              const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'X-API-Key': 'my-secret-key'
                },
                body: JSON.stringify(payload)
              });
              const data = await response.json();
              if (!response.ok) {
                result.textContent = 'Error: ' + (data.error || 'Prediction failed');
                return;
              }
              result.textContent = 'Predicted fare: $' + Number(data.fare_amount).toFixed(2);
            });
          </script>
        </body>
        </html>
        """
    )


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
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "Unauthorized", "status": "failed"}), 401

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
