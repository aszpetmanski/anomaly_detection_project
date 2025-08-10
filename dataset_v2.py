import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of data points
n_samples = 1200

# Generate features with specified distributions
temperature = np.random.normal(loc=18, scale=0.2, size=n_samples)  # Avg 3°C lower
light_exposure = np.random.normal(loc=500, scale=50, size=n_samples)  # Lux
sound_volume = np.random.normal(loc=100, scale=5, size=n_samples)  # dBA
vibration_level = np.random.normal(loc=3, scale=0.5, size=n_samples)  # mm/s RMS

# Create DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'light_exposure': light_exposure,
    'sound_volume': sound_volume,
    'vibration_level': vibration_level
})

# Number of anomalies per type (~1.5% of dataset for each feature)
n_anomalies = int(n_samples * 0.015)
n_low = n_anomalies // 2
n_high = n_anomalies - n_low

# Select anomaly indices per feature
idx_temp = np.random.choice(df.index, size=n_anomalies, replace=False)
idx_light = np.random.choice(df.index.difference(idx_temp), size=n_anomalies, replace=False)
idx_sound = np.random.choice(df.index.difference(idx_temp).difference(idx_light), size=n_anomalies, replace=False)
idx_vibration = np.random.choice(df.index.difference(idx_temp).difference(idx_light).difference(idx_sound), size=n_anomalies, replace=False)

# Inject temperature anomalies (spikes)
df.loc[idx_temp, 'temperature'] = np.random.normal(loc=26, scale=0.5, size=n_anomalies)

# Light exposure anomalies: very low & very high
df.loc[idx_light[:n_low], 'light_exposure'] = np.random.normal(loc=100, scale=20, size=n_low)
df.loc[idx_light[n_low:], 'light_exposure'] = np.random.normal(loc=2000, scale=100, size=n_high)

# Sound volume anomalies: extremely high
df.loc[idx_sound, 'sound_volume'] = np.random.normal(loc=130, scale=5, size=n_anomalies)

# Vibration anomalies: very low or very high
df.loc[idx_vibration[:n_low], 'vibration_level'] = np.random.normal(loc=0.3, scale=0.1, size=n_low)
df.loc[idx_vibration[n_low:], 'vibration_level'] = np.random.normal(loc=9, scale=1, size=n_high)

# Label anomalies
df['anomaly'] = 0
df.loc[df['temperature'] > 25, 'anomaly'] = 1
df.loc[df['light_exposure'] < 200, 'anomaly'] = 1
df.loc[df['light_exposure'] > 1500, 'anomaly'] = 1
df.loc[df['sound_volume'] > 120, 'anomaly'] = 1
df.loc[df['vibration_level'] < 1, 'anomaly'] = 1
df.loc[df['vibration_level'] > 7, 'anomaly'] = 1

# Split for training and streaming
df_train = df.iloc[-200:].reset_index(drop=True)
df_stream = df.iloc[:-200].reset_index(drop=True)

# Save
df_stream.to_csv("streaming_dataset.csv", index=False)
df_train.to_csv("model_training_dataset.csv", index=False)

print("Dataset v2 created with features: temperature, light_exposure, sound_volume, vibration_level")
print(df.head(10))
