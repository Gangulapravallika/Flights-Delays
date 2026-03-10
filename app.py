import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for generating plots

import matplotlib.pyplot as plt
import io
import os
import json
import base64
import subprocess
import warnings
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify

# Suppress specific UserWarnings (e.g., from XGBoost)
warnings.filterwarnings("ignore", category=UserWarning)

# Custom modules
import openweather_data
import preparing_forecast_data
import delay_forecasting

# Flask App
app = Flask(__name__)

# ================================
# Scheduled Background Tasks
# ================================

def fetch_and_process_data():
    print(f"[{datetime.now()}] Starting data fetch and processing...")

    original_dir = os.getcwd()
    os.chdir('flightdelay')  # Navigate to crawler directory

    try:
        subprocess.run(['scrapy', 'crawl', 'DFWFlightsSpider'], check=True)
        print("✔️ Spider completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Spider error: {e}")

    os.chdir(original_dir)

    # Combine flight & weather data, then run delay prediction
    preparing_forecast_data.combine_flight_weather_data()
    delay_forecasting.predict_flight_delays()

def weather_data_fetch_job():
    print(f"[{datetime.now()}] Fetching weather data...")
    openweather_data.fetch_weather_data()

def delete_old_files():
    print(f"[{datetime.now()}] Cleaning up old files...")
    keep_files = {
        "flightdelay/flights_for_display.csv",
        "flightdelay/combined_flight_weather_data.csv",
        "flightdelay/flight_delay_predictions.json"
    }

    for dir_path in [".", "flightdelay"]:
        for file in os.listdir(dir_path):
            full_path = os.path.join(dir_path, file)
            if file.endswith(('.csv', '.json')) and full_path not in keep_files:
                os.remove(full_path)
                print(f"🗑 Deleted: {file}")
    print("✔️ Cleanup complete.")

# ================================
# Helper Functions
# ================================

def load_flight_data(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}")
        return []

def load_flight_delays(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"⚠️ Delay CSV not found: {filename}")
        return pd.DataFrame()

# ================================
# Routes
# ================================

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/flights', methods=['GET'])
def get_flights():
    flights = load_flight_data('flightdelay/flight_delay_predictions.json')
    return jsonify(flights)

@app.route('/flight-statistics')
def flight_statistics():
    predictions = load_flight_data("flightdelay/flight_delay_predictions.json")
    delays = load_flight_delays("flightdelay/flights_for_display.csv")

    predictions_df = pd.DataFrame(predictions)
    delays_df = delays[['flight_number', 'delayed']]

    merged_df = pd.merge(predictions_df, delays_df, on="flight_number", how="inner")
    predicted_delayed_df = merged_df[merged_df['AI Delay Prediction'] == "Yes"]

    # Stats calculations
    predicted_delayed_flights = len(predicted_delayed_df)
    actually_delayed_flights = predicted_delayed_df['delayed'].sum()
    not_delayed_flights = predicted_delayed_flights - actually_delayed_flights
    accuracy_percentage = round((actually_delayed_flights / predicted_delayed_flights * 100), 2) if predicted_delayed_flights else 0
    total_flights_scheduled = len(delays_df)

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(['Actually Delayed', 'Not Actually Delayed'], [actually_delayed_flights, not_delayed_flights], color=['#FF6347', '#87CEFA'])
    ax.set_ylabel('Number of Flights')
    ax.set_title('Predicted vs Actually Delayed Flights')

    # Convert plot to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    statistics = {
        'total_predicted_delayed': predicted_delayed_flights,
        'actually_delayed': actually_delayed_flights,
        'accuracy': accuracy_percentage,
        'total_flights_scheduled': total_flights_scheduled
    }

    accuracy_by_carrier = merged_df.groupby("op_unique_carrier").apply(
        lambda x: pd.Series({
            'predicted_delayed': len(x[x['AI Delay Prediction'] == "Yes"]),
            'actually_delayed': x['delayed'].sum(),
            'accuracy': round((x['delayed'].sum() / len(x[x['AI Delay Prediction'] == "Yes"])) * 100, 2) if len(x[x['AI Delay Prediction'] == "Yes"]) > 0 else 0
        })
    ).reset_index()

    return render_template("flight_statistics.html",
                           statistics=statistics,
                           data=predicted_delayed_df.head(10),
                           bar_chart=img_base64,
                           accuracy_by_carrier=accuracy_by_carrier)

# ================================
# Background Scheduler
# ================================

from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(delete_old_files, 'cron', hour=1, minute=0)
scheduler.add_job(fetch_and_process_data, 'cron', hour=0, minute=30)
scheduler.add_job(weather_data_fetch_job, 'cron', hour=18, minute=0)
scheduler.start()

# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    app.run()
