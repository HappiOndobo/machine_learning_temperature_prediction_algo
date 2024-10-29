from random import shuffle

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import timeseries_dataset_from_array


################### Data Overview and Preprocessiing  #########
#load dataset
fname = "jena_climate_2009_2016.csv"
data = open(fname).readlines()

#split header and data
header = data[0].strip().split(",")
lines = data[1:]

#Extract temperature and other relevant features
temperature = np.zeros(len(lines))
raw_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]

# plot temperature over the entire time range
#plt.figure(figsize=(10, 6))
#plt.plot(temperature)
#plt.title("Temperature Variation Over Time")
#plt.xlabel("time")
#plt.ylabel("Temperature (°C)")
#plt.show()

# Plot temperature change for the first ten days
#Since each record is recorded every 10 minutes,
# there are 144 data points per day (24 hours * 6 records/hour).
# Therefore, for the first ten days, we need to plot the first 1440 data points (10 days * 144 records/day).

ten_days = 10 * 144  # Number of data points for the first 10 days (1440)

plt.figure(figsize=(10, 6))
plt.plot(temperature[:ten_days])
plt.title("Temperature Variation Over the First 10 Days")
plt.xlabel("Time (10-minute intervals)")
plt.ylabel("Temperature (°C)")
plt.show()

#################### Data Normalization ####################

#split dataset into training, validation, and test sets
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

#Normalize the dataset
mean = raw_data[:num_train_samples].mean(axis=0)
std = raw_data[:num_train_samples].std(axis=0)
raw_data_normalized = (raw_data - mean) / std

#Display the normalized temperature data
plt.figure(figure=(10,6))
plt.plot(raw_data[:, 1])
plt.title("Normalized Temperature Variation")
plt.xlabel("Time")
plt.ylabel("Normalized Temperature")
plt.show()

################# Creating Datasets for Time-Series Forecasting ###########

# Parameters for dataset generation
sampling_rate = 6  # 1-hour intervals
sequence_length = 120  # 20-hour sequences
delay = sampling_rate * (sequence_length + 24 - 1)  # Target is 24 hours after the sequence

# Generate training, validation, and test datasets
batch_size = 256

# Training dataset
train_dataset = timeseries_dataset_from_array(
    raw_data_normalized[:-delay],
    targets=temperature[delay:],  # Shifted target by delay
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples
)

# Validation dataset
val_dataset = timeseries_dataset_from_array(
    raw_data_normalized[:-delay],
    targets=temperature[delay:],  # Shifted target by delay
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples
)

# Test dataset
test_dataset = timeseries_dataset_from_array(
    raw_data_normalized[:-delay],
    targets=temperature[delay:],  # Shifted target by delay
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples
)

print("Datasets created successfully!")

