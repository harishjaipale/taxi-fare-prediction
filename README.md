   # Taxi Fare Prediction API

A Flask-based taxi fare prediction service with a polished browser UI, protected prediction endpoint, and deployment-ready setup for Render.

## 🌐 Live demo

Try the deployed app here:

https://taxi-fare-prediction-7.onrender.com/

## ✨ What this project includes

- A live REST API with `/` and `/predict`
- Feature engineering that mirrors the trained model input schema
- A simple browser front end for testing predictions
- API-key protection on `/predict`
- Render deployment configuration via `render.yaml`

## 🚀 Tech stack

- Flask
- pandas / NumPy
- scikit-learn
- joblib
- Gunicorn

## 📋 Prerequisites

- Python 3.10+
- Git
- (Optional) Git LFS for the model file

## ⚙️ Local setup

1. Clone the repository
   ```bash
   git clone https://github.com/harishjaipale/taxi-fare-prediction.git
   cd taxi-fare-prediction
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app locally
   ```bash
   python app.py
   ```

5. Test the API
   - Open http://localhost:5000/
   - Send a POST request to http://localhost:5000/predict
   - Use the header `X-API-Key: my-secret-key` by default

## 🧪 API example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key" \
  -d '{
    "pickup_datetime": "2024-01-15 08:30:00",
    "pickup_latitude": 40.721,
    "pickup_longitude": -73.987,
    "dropoff_latitude": 40.751,
    "dropoff_longitude": -73.97,
    "passenger_count": 1
  }'
```

## 🌐 Deployment

This project is configured for Render with `render.yaml`.

Live deployment: https://taxi-fare-prediction-7.onrender.com/

Recommended Render settings:
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`
- Root Directory: `.`

## 🔐 Notes

- The model file `model.pkl` is expected in the project root.
- If the model is not present locally, run `git lfs pull` to restore the binary file from Git LFS.
- If you want to change the API key, set the `API_KEY` environment variable in your hosting platform.
