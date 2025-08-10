import numpy as np
import pandas as pd
import json

# Set seed for reproducibility
np.random.seed(42)

# Number of data points
n_samples = 1200

# Generate features with specified distributions
temperature = np.random.normal(loc=21,
                               scale=0.2,
                               size=n_samples)  # Stable around 21°C

humidity = np.random.normal(loc=50,
                            scale=1,
                            size=n_samples)  # Stable around 50%

sound_volume = np.random.normal(loc=100,
                                scale=5,
                                size=n_samples)  # Typical industrial noise levels in dBA

# Create DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'sound_volume': sound_volume
})

# Display first few rows
print(df.head(10))

# Random indices for each anomaly type
n_anomalies = int(n_samples * 0.015)  # ~1.5% per type

# Handle rounding to ensure sum == total
n_low = n_anomalies // 2
n_high = n_anomalies - n_low  # ensures total is always correct

idx_temp = np.random.choice(df.index, size=n_anomalies, replace=False)
idx_humidity = np.random.choice(df.index.difference(idx_temp), size=n_anomalies, replace=False)
idx_noise = np.random.choice(df.index.difference(idx_temp).difference(idx_humidity), size=n_anomalies, replace=False)

# Inject temperature anomalies (e.g., sudden heating)
df.loc[idx_temp, 'temperature'] = np.random.normal(loc=26,
                                                   scale=0.5,
                                                   size=n_anomalies)

# Humidity: low + high
df.loc[idx_humidity[:n_low], 'humidity'] = np.random.normal(loc=20,
                                                            scale=2,
                                                            size=n_low)
df.loc[idx_humidity[n_low:], 'humidity'] = np.random.normal(loc=80,
                                                            scale=2,
                                                            size=n_high)

# Inject sound anomalies (extremely high)
df.loc[idx_noise, 'sound_volume'] = np.random.normal(loc=130,
                                                     scale=5,
                                                     size=n_anomalies)

# Save the updated dataset
df.to_csv("factory_sensor_data_with_anomalies.csv", index=False)


#Label anomalies
df['anomaly'] = 0
df.loc[df['temperature'] > 25, 'anomaly'] = 1
df.loc[df['humidity'] < 30, 'anomaly'] = 1
df.loc[df['humidity'] > 70, 'anomaly'] = 1
df.loc[df['sound_volume'] > 120, 'anomaly'] = 1


# Cut the last 200 rows for model training
df_train = df.iloc[-200:].reset_index(drop=True)

# Use first 1000 for streaming only
df_stream = df.iloc[:-200].reset_index(drop=True)

# Save streaming subset to a new file
df_stream.to_csv("streaming_dataset.csv", index=False)
df_train.to_csv("model_training_dataset.csv", index=False)

# Save features dynamically (exclude label column)
feature_list = [col for col in df_train.columns if col != 'anomaly']
with open("features.json", "w") as f:
    json.dump(feature_list, f)

print(f"Features saved: {feature_list}")

df.head(10)  # Display first few rows of the full dataset with anomalies

