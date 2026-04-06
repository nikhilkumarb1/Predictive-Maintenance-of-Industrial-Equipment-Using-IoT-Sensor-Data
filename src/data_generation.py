import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sensor_data(num_records=10000, output_path="data/sensor_data.csv"):
    """
    Generates realistic synthetic IoT sensor data for predictive maintenance.
    The data simulates 5 machines with readings taken every 5 minutes.
    """
    np.random.seed(42)
    
    # Base timestamps
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(num_records)]
    
    # Machine IDs
    machine_ids = np.random.choice(["M_001", "M_002", "M_003", "M_004", "M_005"], num_records)
    
    # Generating normal operating data
    temperature = np.random.normal(60, 5, num_records)      # Normal temp ~60C
    vibration = np.random.normal(1.5, 0.3, num_records)     # Normal vib ~1.5 mm/s
    pressure = np.random.normal(100, 5, num_records)        # Normal psi
    current = np.random.normal(20, 2, num_records)          # Normal amps
    voltage = np.random.normal(220, 5, num_records)         # Normal volts
    rpm = np.random.normal(1500, 50, num_records)           # Normal RPM
    humidity = np.random.normal(45, 10, num_records)        # Normal %
    load = np.random.normal(70, 10, num_records)            # Normal %
    
    failure_status = np.zeros(num_records, dtype=int)
    
    # Introduce anomalies / failures (~10% failure baseline)
    failure_indices = np.random.choice(range(num_records), size=int(0.1 * num_records), replace=False)
    
    for idx in failure_indices:
        failure_type = np.random.choice(['overheating', 'high_vibration', 'pressure_drop', 'power_surge', 'overload'])
        
        if failure_type == 'overheating':
            temperature[idx] += np.random.uniform(20, 50)  # Sudden spike
            vibration[idx] += np.random.uniform(0.5, 1.5)
        elif failure_type == 'high_vibration':
            vibration[idx] += np.random.uniform(2.0, 5.0)
            rpm[idx] -= np.random.uniform(100, 300)
        elif failure_type == 'pressure_drop':
            pressure[idx] -= np.random.uniform(30, 60)
        elif failure_type == 'power_surge':
            current[idx] += np.random.uniform(10, 20)
            voltage[idx] += np.random.uniform(20, 40)
        elif failure_type == 'overload':
            load[idx] += np.random.uniform(20, 40)
            rpm[idx] -= np.random.uniform(200, 400)
            temperature[idx] += np.random.uniform(10, 20)
            
        failure_status[idx] = 1 # Mark as failure

    # Post-process logic to ensure realistic physical boundaries
    pressure = np.maximum(pressure, 0)
    current = np.maximum(current, 0)
    rpm = np.maximum(rpm, 0)
    humidity = np.clip(humidity, 0, 100)
    load = np.clip(load, 0, 100)

    # Compile DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": machine_ids,
        "temperature": np.round(temperature, 2),
        "vibration": np.round(vibration, 4),
        "pressure": np.round(pressure, 2),
        "current": np.round(current, 2),
        "voltage": np.round(voltage, 2),
        "rpm": np.round(rpm, 2),
        "humidity": np.round(humidity, 2),
        "load": np.round(load, 2),
        "failure_status": failure_status
    })

    # Sort data sequentially by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated synthetic dataset with {num_records} records.")
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    generate_sensor_data(output_path="C:/Users/Nikhil/OneDrive/Desktop/Predictive_maintenance/data/sensor_data.csv")
