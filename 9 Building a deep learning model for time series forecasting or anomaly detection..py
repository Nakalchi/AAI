python -m pip install numpy pandas matplotlib scikit-learn tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Step 1: Load the Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

data = pd.read_csv(url, usecols=[1], header=0)
data = data.values.astype("float32")


# Step 2: Visualize the Dataset
plt.plot(data)
plt.title("Airline Passengers Over Time")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()


# Step 3: Normalize the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# Step 4: Prepare Data for LSTM
def create_dataset(dataset, look_back=1):
    X, y = [], []

    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])

    return np.array(X), np.array(y)


look_back = 12

X, y = create_dataset(scaled_data, look_back)

# Reshape data for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))


# Step 5: Split Data into Training and Testing Sets
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Step 6: Build the LSTM Model
model = Sequential([
    LSTM(50, activation="relu", input_shape=(look_back, 1)),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# Step 7: Train the Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# Step 8: Evaluate the Model
loss = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Loss: {loss:.4f}")


# Step 9: Predict Future Values
y_pred = model.predict(X_test)

# Convert Predictions Back to Original Scale
y_pred = scaler.inverse_transform(y_pred)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# Step 10: Plot Actual vs Predicted Values
plt.figure(figsize=(10, 5))

plt.plot(y_test_actual, label="Actual")
plt.plot(y_pred, label="Predicted")

plt.title("Actual vs Predicted Airline Passengers")
plt.xlabel("Time")
plt.ylabel("Passengers")

plt.legend()
plt.show()


# Step 11: Save the Model
model.save("lstm_time_series.keras")

print("\nModel saved as 'lstm_time_series.keras'.")
