import copy
import pickle
import json
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from create_model import manual_model_dense
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

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)

shift_num = 10
df_temp0 = copy.deepcopy(df0)
for i in range(shift_num):
    df0 = pd.concat([df0, df_temp0.groupby('route').shift(periods=i + 1).add_suffix(f'_shifted{i + 1}')], axis=1)

df0.dropna(inplace=True)

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)

pf = PolynomialFeatures(degree=2)
df1 = pf.fit_transform(df0.iloc[:,:-1])
df2 = np.concatenate((df1,df0.iloc[:,-1:].values), axis=1)

x_train, x_test, y_train, y_test = train_test_split(df2[:,:-1], df2[:,-1], test_size=0.0005, shuffle=False)

# =====================================================================================================
model1 = manual_model_dense(x_train)
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es1 = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
history = model1.fit(x_train.reshape((x_train.shape[0], x_train.shape[1], 1)), y_train,
                    validation_data=(x_test, y_test), callbacks=es1, epochs=10000, batch_size=100)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred_train1 = model1.predict(x_train)
y_pred_test1 = model1.predict(x_test)
x_train2 = x_train[(abs(y_pred_train1.reshape(-1)-y_train) >= 300)]
y_train2 = y_train[(abs(y_pred_train1.reshape(-1)-y_train) >= 300)]
# =====================================================================================================
model2 = GradientBoostingRegressor()
model2.fit(x_train2, y_train2)
y_pred_train2 = model1.predict(x_train)
y_pred_test2 = model1.predict(x_test)
x_train3 = np.concatenate((y_pred_train1,y_pred_train2), axis=1)
x_test3 = np.concatenate((y_pred_test1,y_pred_test2), axis=1)
# =====================================================================================================
input_shape = x_train3.shape[1]
input_layer = keras.Input(shape=input_shape)
x = input_layer
x1 = layers.Dense(20, activation="relu")(x)
x2 = layers.Dense(10, activation="relu")(x1)
x3 = layers.Dense(5, activation="relu")(x2)
# x4 = layers.Dense(5, activation="relu")(x3)
# x5 = layers.Dense(5, activation="relu")(x4)
# output_layer1 = layers.Dense(3)(x5)
output_layer = layers.Dense(1)(x3)
model3 = keras.Model(inputs=input_layer, outputs=output_layer)
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es3 = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
history = model3.fit(x_train3, y_train,
                    validation_data=(x_test3, y_test), callbacks=es3, epochs=10000, batch_size=100)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# =====================================================================================================

y_pred = model3.predict(x_test3).reshape(1, -1)
y_actual = y_test.reshape(1, -1)

df_result = pd.DataFrame({'pred': y_pred.reshape(-1),
                          'actual': y_actual.reshape(-1)})
df_result['error'] = df_result['actual'] - df_result['pred']
for i in range(0, 1000, 100):
    print(f'Size of the errors between {i}kg to {i + 100}kg',
          len(abs(df_result['error'])[((abs(df_result['error']) >= i) & (abs(df_result['error']) < i + 100))]) / len(
              df_result['error']))
print(np.mean(np.abs(df_result['error'])))
model1.save('my_model_baggage_model1.h5')
filename = 'model2.sav'
pickle.dump(model2, open(filename, 'wb'))
model3.save('my_model_baggage_model3.h5')
