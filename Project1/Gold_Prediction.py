import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('goldPrice.csv')

# Check for null,missing & remove the entire row from the dataset

if df.isnull().sum().sum() > 0:
    df = df.dropna()

df['Date'] = pd.to_datetime(df['Date'])

# Extract month from the 'Date' column
df['Month'] = df['Date'].dt.month


monthly_stats = df.groupby('Month')['Close'].agg(['max', 'min', 'mean'])

months_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


plt.figure(figsize=(12, 6))

plt.plot(monthly_stats.index, monthly_stats['max'], marker='s', color='green', linestyle='--', linewidth=2, label='Max')
plt.plot(monthly_stats.index, monthly_stats['min'], marker='^', color='red', linestyle='--', linewidth=2, label='Min')
plt.plot(monthly_stats.index, monthly_stats['mean'], marker='x', color='blue', linestyle='-.', linewidth=2, label='Mean')

# Add data point values on each point
for i, (maximum, minimum, mean) in enumerate(zip(monthly_stats['max'], monthly_stats['min'], monthly_stats['mean'])):
    plt.text(monthly_stats.index[i], maximum, f"{maximum:.2f}", ha='right', va='bottom', fontsize=8, color='green')
    plt.text(monthly_stats.index[i], minimum, f"{minimum:.2f}", ha='right', va='bottom', fontsize=8, color='red')
    plt.text(monthly_stats.index[i], mean, f"{mean:.2f}", ha='right', va='bottom', fontsize=8, color='blue')

# date with the maximum, minimum, and mean of the closing prices
plt.title('Monthly Closing Prices Statistics')
plt.xlabel('Month')
plt.ylabel('Closing Price')
plt.xticks(range(1, 13), labels=months_names)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()

# date with the close price over time

df.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], color='green')
plt.title('Gold Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

decomposition = seasonal_decompose(df['Close'], model='additive', period=365)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(10, 6))

plt.subplot(411)
plt.plot(df['Close'], label='Original', color='red')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend, label='Trend', color='red')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(seasonal, label='Seasonal', color='red')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual, label='Residuals',color='red')
plt.legend(loc='upper left')

plt.tight_layout()

plt.show()


from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import Sequential
from keras.layers import LSTM, Dense

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("goldPrice.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

train = df.loc[df.index <= '2023-12-31']
test = df.loc['2024-01-01':]

# ARIMA Model
order_arima = (5, 1, 0)  # Example ARIMA(5,1,0)
arima_model = ARIMA(train['Close'], order=order_arima)
arima_fit = arima_model.fit()
forecast_arima = arima_fit.forecast(steps=len(test))

mae_arima = mean_absolute_error(test['Close'], forecast_arima)
mse_arima = mean_squared_error(test['Close'], forecast_arima)
rmse_arima = np.sqrt(mse_arima)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Close'], label='Actual', color='black')
plt.plot(test.index, forecast_arima[:len(test)], label='ARIMA Prediction', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Forecasting')
plt.legend()
plt.show()

# SARIMA Model
order_sarima = (1, 1, 1)  # Example SARIMA(1,1,1)
seasonal_order_sarima = (1, 1, 1, 12)  # Example seasonal order (1,1,1,12)
sarima_model = SARIMAX(train['Close'], order=order_sarima, seasonal_order=seasonal_order_sarima)
sarima_fit = sarima_model.fit()
forecast_sarima = sarima_fit.forecast(steps=len(test))

mae_sarima = mean_absolute_error(test['Close'], forecast_sarima)
mse_sarima = mean_squared_error(test['Close'], forecast_sarima)
rmse_sarima = np.sqrt(mse_sarima)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Close'], label='Actual', color='black')
plt.plot(test.index, forecast_sarima[:len(test)], label='SARIMA Prediction', color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Forecasting')
plt.legend()
plt.show()

# Holt-Winters Model
hw_model = ExponentialSmoothing(train['Close'], seasonal_periods=12, trend='add', seasonal='add')
hw_fit = hw_model.fit()
forecast_hw = hw_fit.forecast(steps=len(test))

mae_hw = mean_absolute_error(test['Close'], forecast_hw)
mse_hw = mean_squared_error(test['Close'], forecast_hw)
rmse_hw = np.sqrt(mse_hw)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Close'], label='Actual', color='black')
plt.plot(test.index, forecast_hw[:len(test)], label='Holt-Winters Prediction', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Holt-Winters Forecasting')
plt.legend()
plt.show()

# Prophet Model with adjusted changepoint_prior_scale
prophet_model = Prophet()  # Adjust this value as needed
train_prophet = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
test_prophet = test.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model.fit(train_prophet)
future = prophet_model.make_future_dataframe(periods=len(test) + 6)  # Extend future to include 6 additional months
forecast_prophet = prophet_model.predict(future)['yhat'].tail(len(test))

mae_prophet = mean_absolute_error(test_prophet['y'], forecast_prophet)
mse_prophet = mean_squared_error(test_prophet['y'], forecast_prophet)
rmse_prophet = np.sqrt(mse_prophet)

plt.figure(figsize=(12, 6))
plt.plot(test_prophet['ds'], test_prophet['y'], label='Actual', color='black')
plt.plot(test_prophet['ds'], forecast_prophet[:len(test_prophet)], label='Prophet Prediction', color='orange')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prophet Forecasting with Adjusted changepoint_prior_scale')
plt.legend()
plt.show()

# LSTM Model
X_train, y_train = train[['Close', 'Volume']].values, train['Close'].values
X_test, y_test = test[['Close', 'Volume']].values, test['Close'].values
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0)
forecast_lstm = lstm_model.predict(X_test_lstm)

mae_lstm = mean_absolute_error(test['Close'], forecast_lstm)
mse_lstm = mean_squared_error(test['Close'], forecast_lstm)
rmse_lstm = np.sqrt(mse_lstm)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Close'], label='Actual', color='black')
plt.plot(test.index, forecast_lstm[:len(test)], label='LSTM Prediction', color='purple')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('LSTM Forecasting')
plt.legend()
plt.show()

# Print evaluation results
results_df = pd.DataFrame({
    'MAE': [mae_arima, mae_sarima, mae_hw, mae_prophet, mae_lstm],
    'MSE': [mse_arima, mse_sarima, mse_hw, mse_prophet, mse_lstm],
    'RMSE': [rmse_arima, rmse_sarima, rmse_hw, rmse_prophet, rmse_lstm]
}, index=["ARIMA", "SARIMA", "Holt-Winters", "Prophet", "LSTM"])
print(results_df)


# ARIMA Model for Full Data
arima_model_full = ARIMA(df['Close'], order=order_arima)
arima_fit_full = arima_model_full.fit()
forecast_arima_full = arima_fit_full.forecast(steps=len(df) + 6)

mae_arima_full = mean_absolute_error(df['Close'], forecast_arima_full[:len(df)])
mse_arima_full = mean_squared_error(df['Close'], forecast_arima_full[:len(df)])
rmse_arima_full = np.sqrt(mse_arima_full)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual', color='black')
plt.plot(df.index, forecast_arima_full[:len(df)], label='ARIMA Full Prediction', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Forecasting for Full Data')
plt.legend()
plt.show()

# SARIMA Model for Full Data
sarima_model_full = SARIMAX(df['Close'], order=order_sarima, seasonal_order=seasonal_order_sarima)
sarima_fit_full = sarima_model_full.fit()
forecast_sarima_full = sarima_fit_full.forecast(steps=len(df) + 6)

mae_sarima_full = mean_absolute_error(df['Close'], forecast_sarima_full[:len(df)])
mse_sarima_full = mean_squared_error(df['Close'], forecast_sarima_full[:len(df)])
rmse_sarima_full = np.sqrt(mse_sarima_full)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual', color='black')
plt.plot(df.index, forecast_sarima_full[:len(df)], label='SARIMA Full Prediction', color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Forecasting for Full Data')
plt.legend()
plt.show()

# Train-test split for Holt-Winters
train_size_hw = int(len(df) * 0.8)  # 80% of the data for training

train_hw = df.iloc[:train_size_hw]
test_hw = df.iloc[train_size_hw:]

# Holt-Winters Model for Full Data
hw_model_full = ExponentialSmoothing(train_hw['Close'], seasonal_periods=12, trend='add', seasonal='add')
hw_fit_full = hw_model_full.fit()
forecast_hw_full = hw_fit_full.forecast(steps=len(df) + 6)

mae_hw_full = mean_absolute_error(test_hw['Close'], forecast_hw_full[-len(test_hw):])
mse_hw_full = mean_squared_error(test_hw['Close'], forecast_hw_full[-len(test_hw):])
rmse_hw_full = np.sqrt(mse_hw_full)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual', color='black')
plt.plot(df.index, forecast_hw_full[:len(df)], label='Holt-Winters Full Prediction', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Holt-Winters Forecasting for Full Data')
plt.legend()
plt.show()


# Train-test split for Prophet
train_size_prophet = int(len(df) * 0.8)  # 80% of the data for training

train_prophet = df.iloc[:train_size_prophet].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
test_prophet = df.iloc[train_size_prophet:].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# Prophet Model for full data
prophet_model_full = Prophet()
prophet_model_full.fit(train_prophet)
future_full = prophet_model_full.make_future_dataframe(periods=len(test_prophet))  # Extend future to include 6 additional months
forecast_prophet_full = prophet_model_full.predict(future_full)

mae_prophet_full = mean_absolute_error(test_prophet['y'], forecast_prophet_full['yhat'][-len(test_prophet):])
mse_prophet_full = mean_squared_error(test_prophet['y'], forecast_prophet_full['yhat'][-len(test_prophet):])
rmse_prophet_full = np.sqrt(mse_prophet_full)

plt.figure(figsize=(12, 6))
plt.plot(df.index[:len(df)], df['Close'][:len(df)], label='Actual', color='black')
plt.plot(forecast_prophet_full['ds'], forecast_prophet_full['yhat'], label='Prophet Full Prediction', color='orange')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prophet Forecasting for Full Data')
plt.legend()
plt.show()

# Train-test split for LSTM
train_size_lstm = int(len(df) * 0.8)  # 80% of the data for training

train_lstm = df.iloc[:train_size_lstm]
test_lstm = df.iloc[train_size_lstm:]

X_train_lstm, y_train_lstm = train_lstm[['Close', 'Volume']].values, train_lstm['Close'].values
X_test_lstm, y_test_lstm = test_lstm[['Close', 'Volume']].values, test_lstm['Close'].values
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# LSTM Model for Full Data
X_full_lstm = np.reshape(df[['Close', 'Volume']].values, (df.shape[0], 2, 1))
forecast_lstm_full = lstm_model.predict(X_full_lstm)

mae_lstm_full = mean_absolute_error(df['Close'], forecast_lstm_full)
mse_lstm_full = mean_squared_error(df['Close'], forecast_lstm_full)
rmse_lstm_full = np.sqrt(mse_lstm_full)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual', color='black')
plt.plot(df.index, forecast_lstm_full, label='LSTM Full Prediction', color='purple')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('LSTM Forecasting for Full Data')
plt.legend()
plt.show()

# Print evaluation results
results_df = pd.DataFrame({
    'MAE': [mae_arima_full, mae_sarima_full, mae_hw_full, mae_prophet_full, mae_lstm_full],
    'MSE': [mse_arima_full, mse_sarima_full, mse_hw_full, mse_prophet_full, mse_lstm_full],
    'RMSE': [rmse_arima_full, rmse_sarima_full, rmse_hw_full, rmse_prophet_full, rmse_lstm_full]
}, index=["ARIMA_full", "SARIMA_full", "Holt-Winters_full", "Prophet_full", "LSTM_full"])
print(results_df)

# Technical Analytical features
from ta import add_all_ta_features
df = pd.read_csv('goldPrice.csv')
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# feature engineering
# additional technical indicators
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

# Define a simple moving average crossover strategy
def moving_average_crossover_strategy(data, short_window=20, long_window=50):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Generate signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Define Backtesting
def backtest(strategy, data):
    # Initialize portfolio with no holdings
    portfolio = pd.DataFrame(index=data.index).fillna(0.0)

    # Buy 1 unit when signal transitions from 0 to 1
    portfolio['positions'] = strategy['positions']

    portfolio['positions_diff'] = portfolio['positions'].diff()

    # daily portfolio value
    portfolio['holdings'] = data['Close'] * portfolio['positions']

    portfolio['cash'] = 10000 - (portfolio['positions_diff'] * data['Close']).cumsum()

    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    return portfolio

# Apply moving average crossover strategy
signals = moving_average_crossover_strategy(df)
portfolio = backtest(signals, df)

# Reverse Trading Strategy
def reverse_strategy(data):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Reverse the trading logic
    signals['signal'] = np.where(data['Volume'] > data['Volume'].rolling(window=20).mean(), -1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Apply reverse trading strategy
reverse_signals = reverse_strategy(df)
reverse_portfolio = backtest(reverse_signals, df)

plt.figure(figsize=(14, 7))
plt.plot(portfolio['total'], label='Moving Average Crossover Strategy')
plt.plot(reverse_portfolio['total'], label='Reverse Trading Strategy')
plt.title('Backtest Results')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

import matplotlib.dates as mdates

df = pd.read_csv('goldPrice.csv')

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# closing prices over time
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title('Gold Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# investigate the impact of market events on gold prices
# plotting gold closing price during COVID-19 pandemic
covid_start_date = '2019-11-01'
covid_slowdown_date = '2021-12-01'
covid_df = df.loc[covid_start_date:covid_slowdown_date]

plt.figure(figsize=(12, 6))
plt.plot(covid_df['Close'], label='Closing Price', color='red')
plt.title('Gold Closing Prices During COVID-19 Pandemic')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

conflict_start_date = pd.to_datetime('2022-02-24')
latest_date = df.index.max()
latest_date_str = latest_date.strftime('%Y-%m-%d')

# Filter the DataFrame for the period from the invasion date to the latest date available
conflict_df = df.loc[conflict_start_date:latest_date_str]

plt.figure(figsize=(12, 6))
plt.plot(conflict_df.index, conflict_df['Close'], label='Closing Price', color='skyblue')

# Add a marker for the start of the war
plt.scatter(conflict_start_date, conflict_df.loc[conflict_start_date, 'Close'], color='red', label='War Started', s=100)

# Define the slowdown date of the war as a datetime object
war_slowdown_date = pd.to_datetime('2023-12-29')

# Check if the slowdown date falls within the range of dates in the DataFrame before accessing its 'Close' price
if war_slowdown_date in conflict_df.index:
    plt.scatter(war_slowdown_date, conflict_df.loc[war_slowdown_date, 'Close'], color='green', label='War Slowdown', s=100)

plt.title(f'Gold Closing Prices During Russia-Ukraine Conflict (Feb 24, 2022 - {latest_date_str})')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.show()

# assess market sentiment and its influence on short-term and long-term price movements
# plotting correlation between volume and price
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Volume', y='Close', data=df, color='green')
plt.title('Correlation between Volume and Closing Price')
plt.xlabel('Volume')
plt.ylabel('Closing Price (USD)')
plt.grid(True)
plt.show()

from scipy import stats
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('goldPrice.csv')

df['Date'] = pd.to_datetime(df['Date'])
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Information About the Dataset:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
shapiro_test_result, p_value = stats.shapiro(df['Close'])
print("Shapiro-Wilk Test for Normality:")
print("Test statistic:", shapiro_test_result)
print("p-value:", p_value)

pearson_corr = df['Close'].corr(df['Volume'])
print("Pearson Correlation Coefficient between 'Close' and 'Volume':", pearson_corr)

# Test for correlation between 'Close' prices and 'Open' prices
pearson_corr_open = df['Close'].corr(df['Open'])
print("Pearson Correlation Coefficient between 'Close' and 'Open':", pearson_corr_open)

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines',
                         line=dict(color='blue'), name='Closing Price'))

fig.update_layout(title='Gold Closing Prices over Time',
                  xaxis_title='Date',
                  yaxis_title='Closing Price',
                  showlegend=False,
                  xaxis=dict(showgrid=True),
                  yaxis=dict(showgrid=True),
                  width=1100,
                  height=500)

fig.show()

df['Month'] = df['Date'].dt.month

fig = px.box(df, x='Month', y='Close', title='Distribution of Closing Prices by Month')
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Closing Price',
    width=1100,
    height=500)
fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume'], mode='lines',
                         line=dict(color='green'), name='Volume'))

fig.update_layout(title='Volume of Gold Trading over Time',
                  xaxis_title='Date',
                  yaxis_title='Volume',
                  showlegend=False,
                  xaxis=dict(showgrid=True),
                  yaxis=dict(showgrid=True),
                  width=1100, # specify the width of the figure
                  height=500) # specify the height of the figure)

fig.show()

# Plotting a scatter plot to visualize the relationship between Closing and Opening Prices
plt.figure(figsize=(12, 6))
plt.scatter(df['Open'], df['Close'], color='red', alpha=0.5)
plt.title('Scatter Plot: Opening vs Closing Prices')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.grid(True)
plt.show()

# Plotting histograms to visualize the distributions of 'Close' and 'Volume'
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Close'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Closing Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['Volume'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Volume')
plt.xlabel('Volume')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()