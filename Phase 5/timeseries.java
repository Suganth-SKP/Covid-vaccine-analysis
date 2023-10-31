import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
# Load your time series data
data = pd.read_csv('country_vaccinations_by_manufacturer.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Check for stationarity
result = adfuller(data['total_vaccinations'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If the data is non-stationary, difference it to make it stationary
if result[1] > 0.05:
    data['total_vaccinations'] = data['total_vaccinations'].diff().dropna()

# Plot the ACF and PACF to determine model orders (p and q)
plot_acf(data['total_vaccinations'])
plot_pacf(data['total_vaccinations'])
plt.show()

# Fit the ARIMA model
model = ARIMA(data['total_vaccinations'], order=(1, 1, 1))  # Adjust p, d, and q as needed
model_fit = model.fit()

# Forecast future values
forecast_steps = 10  # Number of steps to forecast
forecast = model_fit.forecast(steps=forecast_steps)

# Create a date range for the forecasted values
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1)

# Plot the original data and forecasted values
plt.plot(data['total_vaccinations'], label='Original Data')
plt.plot(forecast_index[1:], forecast, label='Forecast', color='red')
plt.legend()
plt.show()
