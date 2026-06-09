# Taxi Fare Prediction API

A machine learning-powered REST API that predicts taxi fares based on pickup/dropoff coordinates, date/time, and passenger count. Built with **Flask**, **scikit-learn**, and **Haversine distance calculations**.

## 🚀 Features

- **Real-time Prediction**: Estimates fare amounts instantly via a POST request.
- **Feature Engineering**: Automatically extracts time-based (hour, weekday) and spatial features (distance, bearing) from raw coordinates.
- **Robust Model**: Uses a Tuned Random Forest regressor trained on the NYC Taxi dataset.
- **Scalable**: Ready for deployment with production-grade dependencies (Gunicorn).

## 📋 Prerequisites

- Python 3.8+
- Git (for cloning the repository)
- (Optional) Git LFS (if you need to regenerate the model locally)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harishjaipale/taxi-fare-prediction.git
   cd taxi-fare-prediction
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Model**: Ensure `model.pkl` exists in the root directory. If you see a pointer file instead of the binary, run:
   ```bash
   git lfs pull
   ```
