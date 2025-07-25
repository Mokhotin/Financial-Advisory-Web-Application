# Importing necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Applying a clean plot style
plt.style.use('fivethirtyeight')

# =======================================================
# 1. Fetching Stock Data (Bitcoin)
# =======================================================
end = datetime.now()
start = datetime(end.year - 4, end.month, end.day)  # Set the date range (last 4 years)
stock = 'BTC-USD'  # Bitcoin price in USD
stock_data = yf.download(stock, start=start, end=end)

# =======================================================
# 2. Fetching Fear & Greed Index Data
# =======================================================
url = "https://api.alternative.me/fng/?limit=0"
response = requests.get(url)
data = response.json()

# Convert API data to DataFrame
sentiment_data = pd.DataFrame(data['data'])

# Convert timestamp to readable date format
sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'], unit='s')

# Set timestamp as index and rename columns
sentiment_data.set_index('timestamp', inplace=True)
sentiment_data.rename(columns={'value': 'Fear & Greed Index', 'value_classification': 'Sentiment'}, inplace=True)

# Keep only necessary columns
sentiment_data = sentiment_data[['Fear & Greed Index', 'Sentiment']]

# Sort and filter data to match the stock data range
sentiment_data.index.name = 'Date'
sentiment_data = sentiment_data.sort_index(ascending=True)
filtered_sentiment_data = sentiment_data[
    (sentiment_data.index >= start) & (sentiment_data.index <= end)
]

# Ensure indices are in the same format
stock_data.index = pd.to_datetime(stock_data.index).date
filtered_sentiment_data.index = pd.to_datetime(filtered_sentiment_data.index).date

# =======================================================
# 3. Exploratory Analysis
# =======================================================
# Displaying basic information and statistics
print("Stock Data Overview:")
print(stock_data.describe().T)
print("\nSentiment Data Overview:")
print(sentiment_data.describe().T)

print("\nStock Data Info:")
stock_data.info()
print("\nSentiment Data Info:")
sentiment_data.info()

print("\nStock Data Columns:", stock_data.columns)
print("\nSentiment Data Columns:", sentiment_data.columns)

# =======================================================
# 4. Visualization of Bitcoin Close Prices
# =======================================================
# Extract and plot closing price
closing_price = stock_data[['Close']]  # Extracting close price
closing_price['Close']

plt.figure(figsize=(15, 6))
plt.plot(closing_price.index, closing_price['Close'], label='Close Price', color='blue', linewidth=2)
plt.title("Close Price of Bitcoin Over Time", fontsize=16)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# =======================================================
# 5. Moving Averages
# =======================================================
# Calculate moving averages
closing_price['MA_365'] = closing_price['Close'].rolling(window=365).mean()
closing_price['MA_100'] = closing_price['Close'].rolling(window=100).mean()

# Plot close prices with moving averages
plt.figure(figsize=(15, 6))
plt.plot(closing_price.index, closing_price['Close'], label='Close Price', color='blue', linewidth=2)
plt.plot(closing_price.index, closing_price['MA_365'], label='365 Days Moving Average', color='red', linestyle="--", linewidth=2)
plt.plot(closing_price.index, closing_price['MA_100'], label='100 Days Moving Average', color='green', linestyle="--", linewidth=2)
plt.title("Close Price with Moving Averages", fontsize=16)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# =======================================================
# 6. Fear & Greed Index Visualization
# =======================================================
plt.figure(figsize=(12, 6))
plt.plot(filtered_sentiment_data.index, filtered_sentiment_data['Fear & Greed Index'], label='Fear & Greed Index', color='purple')
plt.title("Fear & Greed Index Over Time", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Fear & Greed Index", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# =======================================================
# 7. Scaling the Data
# =======================================================
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale closing prices
stock_scaled_data = scaler.fit_transform(closing_price[['Close']].dropna())

# Scale Fear & Greed Index
filtered_sentiment_data['Fear & Greed Index'] = filtered_sentiment_data['Fear & Greed Index'].astype(float)
fear_greed_scaled_data = scaler.fit_transform(filtered_sentiment_data[['Fear & Greed Index']])

# Combine scaled features into a single DataFrame
combined_data = pd.DataFrame({
    'Scaled Close Price': stock_scaled_data.flatten(),
    'Scaled Fear & Greed Index': fear_greed_scaled_data.flatten()
})

# Convert combined data into a NumPy array
combined_array = combined_data.values

# =======================================================
# 8. Preparing Data for LSTM
# =======================================================
x_data = []
y_data = []
base_days = 100  # Number of days for each input sequence

for i in range(base_days, len(combined_array)):
    x_data.append(combined_array[i - base_days:i])
    y_data.append(combined_array[i, 0])  # Target: closing price

x_data = np.array(x_data)
y_data = np.array(y_data)

# Split into training and test sets
train_size = int(len(x_data) * 0.9)
x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]

# =======================================================
# 9. Building the LSTM Model
# =======================================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 2)),  # Input shape matches 2 features
    LSTM(64, return_sequences=False),
    Dense(25),
    Dense(1)  # Single output: predicted scaled close price
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=5, epochs=10)

# =======================================================
# 10. Making Predictions
# =======================================================
predictions = model.predict(x_test)

# Inverse transform predictions and actual values
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions vs actual values
plotting_data = pd.DataFrame({
    'Original': inv_y_test.flatten(),
    'Prediction': inv_predictions.flatten()
}, index=closing_price.index[train_size + base_days:])

plt.figure(figsize=(15, 6))
plt.plot(plotting_data.index, plotting_data['Original'], label='Original', color='blue', linewidth=2)
plt.plot(plotting_data.index, plotting_data['Prediction'], label='Prediction', color='red', linewidth=2)
plt.title("Prediction vs Actual Close Price", fontsize=16)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# =======================================================
# 11. Predicting Future Prices
# =======================================================
last_100 = combined_array[-100:].reshape(1, -1, 2)
future_predictions = []

for _ in range(10):
    next_day = model.predict(last_100)
    future_predictions.append(scaler.inverse_transform(next_day[:, 0].reshape(-1, 1)))
    last_100 = np.append(last_100[:, 1:, :], next_day.reshape(1, 1, -1), axis=1)

future_predictions = np.array(future_predictions).flatten()

# Plot future predictions
plt.figure(figsize=(15, 6))
plt.plot(range(1, 11), future_predictions, marker="o", label="Predicted Future Prices", color="purple", linewidth=2)
for i, val in enumerate(future_predictions):
    plt.text(i + 1, val, f"{val:.2f}", fontsize=10, ha="center", va="bottom", color="black")
plt.title("Future Close Prices for 10 Days", fontsize=16)
plt.xlabel("Day Ahead", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()
