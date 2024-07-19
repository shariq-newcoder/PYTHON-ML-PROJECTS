import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False)

# Combine 'Date' and 'Time' columns to create a 'datetime' column
dataset['datetime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'], format='%d/%m/%Y %H:%M:%S')

# Set 'datetime' as the index
dataset.set_index('datetime', inplace=True)

# Drop the original 'Date' and 'Time' columns
dataset.drop(columns=['Date', 'Time'], inplace=True)

# Mark all missing values
dataset.replace('?', np.nan, inplace=True)

# Convert all columns to numeric, forcing errors to NaN
for col in dataset.columns:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# Handle missing values
# You can fill missing values using forward fill, backward fill, or interpolation
dataset.ffill(inplace=True)  # Forward fill
dataset.bfill(inplace=True)  # Backward fill

# Alternatively, use interpolation
# dataset.interpolate(method='time', inplace=True)

# Handle outliers if necessary
# One common approach is to remove values that are significantly different from the mean
# Here is an example of removing outliers that are more than 3 standard deviations from the mean
for col in dataset.columns:
    mean = dataset[col].mean()
    std = dataset[col].std()
    outliers = (dataset[col] - mean).abs() > 3 * std
    dataset.loc[outliers, col] = np.nan

# Fill the outliers with interpolation again if needed
dataset.interpolate(method='time', inplace=True)

# Add a column for the remainder of sub metering
values = dataset.values
dataset['sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])

# Save the cleaned and preprocessed dataset
dataset.to_csv('household_power_consumption_cleaned.csv')

print("Data preprocessing complete. Cleaned data saved to 'household_power_consumption_cleaned.csv'.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
dataset = pd.read_csv('household_power_consumption_cleaned.csv', index_col='datetime', parse_dates=True)

# Conduct EDA
# 1. Basic statistical summary
print(dataset.describe())

# 2. Plotting Global Active Power over time to observe patterns and trends
plt.figure(figsize=(14, 7))
plt.plot(dataset.index, dataset['Global_active_power'], label='Global Active Power', color='blue')
plt.title('Global Active Power Over Time')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend(loc='upper right')  # Specify legend location
plt.show()

# 3. Plotting a histogram of Global Active Power to observe the distribution
plt.figure(figsize=(10, 5))
sns.histplot(dataset['Global_active_power'], bins=50, kde=True, color='blue')
plt.title('Distribution of Global Active Power')
plt.xlabel('Global Active Power (kilowatts)')
plt.ylabel('Frequency')
plt.show()

# 4. Plotting correlation matrix to visualize relationships between features
correlation_matrix = dataset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# 5. Seasonal decomposition of time series
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(dataset['Global_active_power'], model='additive', period=24*60)
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()

# 6. Pair plot to visualize relationships between different sub_metering values
sns.pairplot(dataset[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'sub_metering_4']])
plt.show()

# 7. Plotting the time series for different sub_metering values to observe patterns
plt.figure(figsize=(14, 7))
plt.plot(dataset.index, dataset['Sub_metering_1'], label='Sub Metering 1', color='blue')
plt.plot(dataset.index, dataset['Sub_metering_2'], label='Sub Metering 2', color='orange')
plt.plot(dataset.index, dataset['Sub_metering_3'], label='Sub Metering 3', color='green')
plt.plot(dataset.index, dataset['sub_metering_4'], label='Sub Metering 4', color='red')
plt.title('Sub Metering Over Time')
plt.xlabel('Time')
plt.ylabel('Energy sub metering')
plt.legend(loc='upper right')  # Specify legend location
plt.show()

# TIME SERIES MODELLINGS PREPROCESSING
import pandas as pd

# Load the cleaned dataset
dataset = pd.read_csv('household_power_consumption_cleaned.csv', index_col='datetime', parse_dates=True)

# Use only the 'Global_active_power' column for time series forecasting
data = dataset['Global_active_power']

# Resample the data to hourly averages to reduce noise
data = data.resample('h').mean()

# Handle any remaining missing values by interpolation
data.interpolate(method='time', inplace=True)

# Split the data into training and test sets (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Print the size of the train and test sets
print(f'Train size: {len(train)}, Test size: {len(test)}')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ARIMA Model
arima_model = ARIMA(train, order=(5, 1, 0))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=len(test))

# Evaluate ARIMA
arima_mse = mean_squared_error(test, arima_forecast)
print(f'ARIMA MSE: {arima_mse}')

# SARIMA Model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
sarima_model_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_model_fit.forecast(steps=len(test))

# Evaluate SARIMA
sarima_mse = mean_squared_error(test, sarima_forecast)
print(f'SARIMA MSE: {sarima_mse}')

# Plot the results
# plt.figure(figsize=(14, 7))
# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
# plt.plot(test.index, sarima_forecast, label='SARIMA Forecast')
# plt.legend()
# plt.show()

# from tensorflow.keras.models import Sequential
from keras import Sequential
from tensorflow import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

train_scaled = scaled_data[:train_size]
test_scaled = scaled_data[train_size:]

# Function to create a dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 24
X_train, Y_train = create_dataset(train_scaled, look_back)
X_test, Y_test = create_dataset(test_scaled, look_back)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], look_back, 1)
X_test = X_test.reshape(X_test.shape[0], look_back, 1)

# Build and train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=2)

# Forecast with LSTM
lstm_forecast = lstm_model.predict(X_test)
lstm_forecast = scaler.inverse_transform(lstm_forecast)

# Evaluate LSTM
lstm_mse = mean_squared_error(test[look_back + 1:], lstm_forecast)
print(f'LSTM MSE: {lstm_mse}')

# Forecast future consumption (e.g., next 7 days)
future_steps = 7

# ARIMA forecast
arima_forecast = arima_model_fit.forecast(steps=future_steps)

# SARIMA forecast
sarima_forecast = sarima_model_fit.forecast(steps=future_steps)

# LSTM forecast
last_values = train_scaled[-look_back:]
lstm_input = last_values.reshape((1, look_back, 1))
lstm_forecast_scaled = lstm_model.predict(lstm_input)
lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled)

# Create a future datetime index
future_dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq='H')[1:]


# Plot the results
# plt.figure(figsize=(14, 7))
# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.plot(test.index[look_back + 1:], lstm_forecast, label='LSTM Forecast')
# plt.legend()
# plt.show()

# Plot future forecasts
plt.figure(figsize=(14, 7))
plt.plot(data.index[-48:], data.values[-48:], label='Actual')
plt.plot(future_dates, arima_forecast, label='ARIMA Forecast')
plt.plot(future_dates, sarima_forecast, label='SARIMA Forecast')
plt.plot(future_dates, lstm_forecast.flatten(), label='LSTM Forecast')
plt.legend(loc='upper left')
plt.title('Future Electricity Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.show()

# Plot forecasts
plt.figure(figsize=(14, 7))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast')
plt.plot(test.index[look_back + 1:], lstm_forecast, label='LSTM Forecast')
plt.legend(loc='upper left')
plt.title('Forecasting Results')
plt.show()

import pandas as pd
import numpy as np

# Load the cleaned dataset
dataset = pd.read_csv('household_power_consumption_cleaned.csv', index_col='datetime', parse_dates=True)

# Feature engineering: Adding time-based features
dataset['hour'] = dataset.index.hour
dataset['day_of_week'] = dataset.index.dayofweek
dataset['month'] = dataset.index.month

# Lagged variables (previous day, previous week)
dataset['lag_24h'] = dataset['Global_active_power'].shift(24)
dataset['lag_168h'] = dataset['Global_active_power'].shift(168)

# Rolling statistics (3-hour window)
dataset['rolling_mean'] = dataset['Global_active_power'].rolling(window=3).mean()
dataset['rolling_std'] = dataset['Global_active_power'].rolling(window=3).std()

# Dummy variables for holidays (hypothetical example)
holidays = ['2024-01-01', '2024-12-25']  # Example holidays
dataset['is_holiday'] = dataset.index.date.astype('str').isin(holidays).astype(int)

# Drop rows with NaN values resulting from lagged variables
dataset.dropna(inplace=True)

# Print the updated dataset with engineered features
print(dataset.head())

# Save the updated dataset with engineered features
dataset.to_csv('household_power_consumption_featured.csv')


