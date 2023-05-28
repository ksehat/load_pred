import copy
import numpy as np
import requests
from functions import api_token_handler
from load_pred import mean_load_pred
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
from create_model import manual_model_conv2d
import pickle
import json
from functions import api_token_handler

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load your data into a DataFrame
token = api_token_handler()
df0 = pd.DataFrame(
    json.loads(requests.get(url='http://192.168.115.10:8083/api/FlightBaggageEstimate/GetAllPastFlightsBaggage',
                            headers={'Authorization': f'Bearer {token}',
                                     'Content-type': 'application/json',
                                     }
                            ).text)['getAllPastFlightsBaggageResponseItemViewModels'])

df0.drop('pkFlightInformation', axis=1, inplace=True)

df0['baggage'] = df0['baggage'].str.split('/', expand=True)[1]
df0['baggage'] = df0['baggage'].str.split(' ', expand=True)[0]
df0['baggage'] = df0['baggage'].astype(float)

df0['year'] = np.array(pd.DatetimeIndex(df0['departure']).year)
df0['month'] = np.array(pd.DatetimeIndex(df0['departure']).month)
df0['day'] = np.array(pd.DatetimeIndex(df0['departure']).day)
df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['departure']).dayofweek)
df0['hour'] = np.array(pd.DatetimeIndex(df0['departure']).hour)
df0['is_holiday'][(np.array(pd.DatetimeIndex(df0['departure']).day_name()) == 'Friday')] = 1
df0['is_holiday'][(np.array(pd.DatetimeIndex(df0['departure']).day_name()) == 'Thursday')] = 1

df0['departure'] = pd.to_datetime(df0['departure'])
df0.sort_values(by='departure', inplace=True)
df0.reset_index(drop=True, inplace=True)

holidays = df0.loc[df0['is_holiday'] == 1, 'departure']
df0['days_until_holiday'] = holidays.reindex(df0.index, method='bfill').dt.date - df0['departure'].dt.date
df0['days_until_holiday'] = pd.to_timedelta(df0['days_until_holiday']).dt.days

le_route = LabelEncoder()
df0['route'] = le_route.fit_transform(df0['route'])

with open('label_encoder_baggage.pkl', 'wb') as f:
    pickle.dump(le_route, f)

df0.drop(['departure', 'paxWeight', 'payLoad'], inplace=True, axis=1)

shift_num = 10
df_temp0 = copy.deepcopy(df0)
for i in range(shift_num):
    df0 = pd.concat([df0, df_temp0.groupby('route').shift(periods=i + 1).add_suffix(f'_shifted{i + 1}')], axis=1)

df0.dropna(inplace=True)

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)
df0.reset_index(inplace=True, drop=True)

# Define the window size and window step
window_size = 50
window_step = 1

data = df0.values

# Create the windowed data
shape = ((data.shape[0] - window_size) // window_step + 1, window_size, data.shape[1])
strides = (window_step * data.strides[0], data.strides[0], data.strides[1])
windowed_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

# Add an extra dimension to the data to make it compatible with Conv2D
windowed_data = windowed_data[..., np.newaxis]

test_size = 1000
x_train = windowed_data[:-test_size, :, :-1]
x_test = windowed_data[-test_size:, :, :-1]
y_train = windowed_data[:-test_size, -1, -1]
y_test = windowed_data[-test_size:, -1, -1]

model = manual_model_conv2d(x_train)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test), callbacks=es, epochs=10000, batch_size=100)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# model.fit(x_train, y_train)
y_pred = model.predict(x_test).reshape(1, -1)
y_actual = y_test.values.reshape(1, -1)

df_result = pd.DataFrame({'pred': y_pred.reshape(-1),
                          'actual': y_actual.reshape(-1)})
df_result['error'] = df_result['actual'] - df_result['pred']
for i in range(0, 1000, 100):
    print(f'Size of the errors between {i}kg to {i + 100}kg',
          len(abs(df_result['error'])[((abs(df_result['error']) >= i) & (abs(df_result['error']) < i + 100))]) / len(
              df_result['error']))
print(np.mean(np.abs(df_result['error'])))
model.save('my_model_baggage_conv2d.h5')
