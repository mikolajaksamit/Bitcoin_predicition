#The project focuses on utilizing machine learning techniques
#to predict Bitcoin prices. Leveraging LSTM networks,
#the model has been trained on historical data to forecast future prices with high precision.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from alpha_vantage.cryptocurrencies import CryptoCurrencies

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

cc = CryptoCurrencies(key='YOUR_API_KEY', output_format='pandas')

crypto_currency = 'BTC'
market = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()
data, meta_data = cc.get_digital_currency_daily(symbol=crypto_currency, market=market)

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['1a. open (USD)'].values.reshape(-1,1))

prediction_day = 60

x_train, y_train = [], []

for x in range(prediction_day,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_day:x, 0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#create neural network

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train, epochs=25, batch_size=32)

#testing the model

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data, _ = cc.get_digital_currency_daily(symbol=crypto_currency, market=market)
actual_price = test_data['1a. open (USD)'].values

total_data = pd.concat((data['1a. open (USD)'], test_data['1a. open (USD)']), axis=0)

model_inputs = total_data[len(total_data)-len(test_data)-prediction_day:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_day, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_day:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)

prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_price,color='black', label='Actual')
plt.plot(prediction_prices, color='green', linestyle='--', alpha=0.5, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('time')
plt.ylabel('price')
plt.legend(loc='upper left')
plt.show()


#%%
