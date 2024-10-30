import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
import pandas as pd

# Set of initial parameters for data generation
np.random.seed(42)
data_length = 1000  # length of data stream
seasonal_amplitude = 10  # seasonal amplitude for generating sine wave
noise_level = 2  # noise level for random variations
drift_rate = 0.02  # drift rate to simulate gradual trend
anomaly_percentage = 0.05  # % of data points to be anomalies


# Function to simulate continuous data stream with injected anomalies
def simulate_data_stream_with_anomalies(length, seasonal_amplitude, noise_level, drift_rate, anomaly_percentage):
    seasonal = seasonal_amplitude * np.sin(np.linspace(0, 10 * np.pi, length))
    noise = np.random.normal(0, noise_level, length)
    drift = np.linspace(0, drift_rate * length, length)
    data_stream = seasonal + noise + drift

    # Calculate number of anomalies (5%) and inject them randomly in the data stream
    num_anomalies = int(length * anomaly_percentage)
    anomaly_indices = np.random.choice(length, num_anomalies, replace=False)

    for idx in anomaly_indices:
        # Introduce a large deviation (positive or negative)
        anomaly_value = data_stream[idx] + np.random.choice([-1, 1]) * np.random.uniform(15, 25)
        data_stream[idx] = anomaly_value

    return data_stream, anomaly_indices


# Function for anomaly detection using EWMA (Exponentially Weighted Moving Average) with Z-score
def ewma_z_score_anomaly_detection(data, alpha=0.18, threshold=1.8, rolling_window_size=50):
    anomalies = []
    ewma_values = []

    # Initialize EWMA with the first value
    ewma = data[0]
    ewma_values.append(ewma)

    for i in range(len(data)):
        if i > 0:
            ewma = alpha * data[i] + (1 - alpha) * ewma
            ewma_values.append(ewma)

        if i < rolling_window_size:
            rolling_std = np.std(data[:i + 1])
        else:
            rolling_std = np.std(data[i - rolling_window_size + 1:i + 1])

        z_score = (data[i] - ewma) / rolling_std if rolling_std > 0 else 0
        is_anomaly = abs(z_score) > threshold
        anomalies.append(is_anomaly)

    return anomalies


# Function to print model performance score
def print_performance_score(detected_anomalies, actual_anomalies):
    detected_set = set(np.where(detected_anomalies)[0])
    actual_set = set(actual_anomalies)
    true_positives = detected_set.intersection(actual_set)
    score = len(true_positives) / len(actual_set) if actual_set else 0
    print(f"Model Performance: Detected {len(true_positives)}/{len(actual_set)} actual anomalies (Score: {score:.2f})")
    return score


# Real-time plot of the data stream with anomaly detection using Matplotlib
def plot_real_time_full_history(data_stream, anomalies_detected, actual_anomalies, run_number):
    """Real-time plot of the data stream with anomaly detection using Matplotlib."""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 6))

    all_x = []
    all_y = []
    detected_anomaly_x = []
    detected_anomaly_y = []
    actual_anomaly_x = []
    actual_anomaly_y = []

    line_data, = ax.plot([], [], label="Data Stream", color='blue')
    scatter_detected, = ax.plot([], [], 'ro', label="Detected Anomalies", markersize=8)
    scatter_actual, = ax.plot([], [], 'bo', label="Actual Anomalies", markersize=3)

    ax.set_xlim(0, len(data_stream))
    ax.set_ylim(np.min(data_stream) - 10, np.max(data_stream) + 10)
    ax.set_title(f"Anomaly Detection Algorithm with {run_number:.2%} detection accuracy")
    ax.legend()

    def update(frame):
        all_x.append(frame)
        all_y.append(data_stream[frame])

        if anomalies_detected[frame]:
            detected_anomaly_x.append(frame)
            detected_anomaly_y.append(data_stream[frame])

        if frame in actual_anomalies:
            actual_anomaly_x.append(frame)
            actual_anomaly_y.append(data_stream[frame])

        line_data.set_data(all_x, all_y)
        scatter_detected.set_data(detected_anomaly_x, detected_anomaly_y)
        scatter_actual.set_data(actual_anomaly_x, actual_anomaly_y)

        return line_data, scatter_detected, scatter_actual

    ani = FuncAnimation(fig, update, frames=len(data_stream), blit=True, repeat=False)

    plt.show()  # Show the plot
    plt.ioff()  # Turn off interactive mode

    # Wait until the figure is closed to exit
    plt.waitforbuttonpress()  # Keep the window open until a button is pressed


# Main execution
def main():
    data_stream, actual_anomalies = simulate_data_stream_with_anomalies(data_length, seasonal_amplitude, noise_level,
                                                                        drift_rate, anomaly_percentage)
    anomalies_detected = ewma_z_score_anomaly_detection(data_stream)
    anomaly_score = print_performance_score(anomalies_detected, actual_anomalies)
    plot_real_time_full_history(data_stream, anomalies_detected, actual_anomalies, run_number=anomaly_score)


# Execute the main function
if __name__ == "__main__":
    main()
