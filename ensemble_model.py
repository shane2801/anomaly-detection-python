import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Set parameters
np.random.seed(42)
data_length = 1000


# Generate example data scenarios
def generate_scenario_data(length, seasonal_amplitude, noise_level, drift_rate, anomaly_percentage):
    """Simulate different data streams with seasonality, noise, drift, and anomalies."""
    seasonal = seasonal_amplitude * np.sin(np.linspace(0, 10 * np.pi, length))
    noise = np.random.normal(0, noise_level, length)
    drift = np.linspace(0, drift_rate * length, length)
    data_stream = seasonal + noise + drift

    # Inject anomalies at random indices
    num_anomalies = int(length * anomaly_percentage)
    anomaly_indices = np.random.choice(length, num_anomalies, replace=False)
    for idx in anomaly_indices:
        data_stream[idx] += np.random.choice([-1, 1]) * np.random.uniform(15, 25)
    return data_stream, anomaly_indices


# Example scenarios
scenarios = [generate_scenario_data(data_length, 10, 2, 0.01, 0.05) for _ in range(3)]  # 3 scenarios


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


# Updated Ensemble Method incorporating LOF and adaptive sensitivity tuning
def ensemble_anomaly_detection(data, alpha, ewma_threshold, ewma_window, iforest_contamination, stl_period,
                               stl_threshold, lof_n_neighbors):
    """Ensemble approach with LOF, STL residuals, Isolation Forest, and EWMA with adaptive thresholding."""
    # EWMA and Z-score Detector
    ewma_anomalies = ewma_z_score_anomaly_detection(data, alpha, ewma_threshold, ewma_window)

    # Isolation Forest
    iforest = IsolationForest(contamination=iforest_contamination, random_state=42)
    iforest_anomalies = iforest.fit_predict(data.reshape(-1, 1))
    iforest_anomalies = iforest_anomalies == -1  # Convert -1 to True anomalies

    # STL Residual Analysis
    stl = STL(data, period=stl_period, robust=True)
    stl_res = stl.fit()
    residuals = stl_res.resid
    stl_anomalies = np.abs(zscore(residuals)) > stl_threshold

    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=iforest_contamination)
    lof_anomalies = lof.fit_predict(data.reshape(-1, 1))
    lof_anomalies = lof_anomalies == -1  # Convert -1 to True anomalies

    # Adaptive Voting Mechanism
    detected_anomalies = []
    for i in range(len(data)):
        anomaly_votes = sum([ewma_anomalies[i], iforest_anomalies[i], stl_anomalies[i], lof_anomalies[i]])
        detected_anomalies.append(anomaly_votes >= 2)  # Adjust majority threshold as necessary

    return detected_anomalies


# Function to calculate accuracy
def calculate_accuracy(detected_anomalies, actual_anomalies):
    tp = sum([1 for i in actual_anomalies if detected_anomalies[i]])
    fp = sum([1 for i, x in enumerate(detected_anomalies) if x and i not in actual_anomalies])
    return tp / (tp + fp + 1e-5)


# Function to run the detection with best parameters
def run_best_parameter_detection(scenarios):
    # Best parameters for each scenario
    best_params = [
        (0.3631106948886222, 1.436683537180478, 22, 0.07482577111153024, 32, 2.284759808932835, 8),  # Scenario 1
        (0.41238962199864787, 1.2943692452782096, 19, 0.05146497215517092, 38, 2.156656211256941, 5),  # Scenario 2
        (0.3848744052107269, 1.4144088267019868, 26, 0.053289910889761954, 26, 2.0875714768971996, 5)  # Scenario 3
    ]

    # Create subplots for all scenarios
    fig, axs = plt.subplots(len(scenarios), figsize=(12, 6 * len(scenarios)))

    for scenario_num, (data_stream, actual_anomalies) in enumerate(scenarios):
        # Get the best parameters for this scenario
        params = best_params[scenario_num]

        # Run ensemble detection with best parameters
        detected_anomalies = ensemble_anomaly_detection(data_stream, *params)

        # Calculate accuracy
        accuracy = calculate_accuracy(detected_anomalies, actual_anomalies)

        # Plot the results for each scenario
        axs[scenario_num].plot(data_stream, label="Data Stream")
        axs[scenario_num].scatter(actual_anomalies, data_stream[actual_anomalies], color='blue', marker='o',
                                  label="Actual Anomalies")
        axs[scenario_num].scatter(np.where(detected_anomalies)[0], data_stream[np.where(detected_anomalies)],
                                  color='red', marker='x',
                                  label="Detected Anomalies")
        axs[scenario_num].set_title(
            f"Scenario {scenario_num + 1} - Anomaly Detection Results (Accuracy: {accuracy:.2f})")
        axs[scenario_num].legend()

    # Adjust layout and save all figures
    plt.tight_layout()
    plt.savefig("All_Scenarios_Anomaly_Detection.png")
    plt.show()


# Execute main function for all scenarios
run_best_parameter_detection(scenarios)
