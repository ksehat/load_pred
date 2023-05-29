import copy
import numpy as np
import pandas as pd
import pickle
import json
import requests
from sklearn.ensemble import GradientBoostingRegressor as model_sklearn
import tensorflow as tf
from keras.callbacks import EarlyStopping
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

with open('label_encoder_baggage.pkl', 'rb') as f:
    le_route = pickle.load(f)
df0['route'] = le_route.fit_transform(df0['route'])

df0.drop(['departure', 'paxWeight', 'payLoad'], inplace=True, axis=1)

shift_num = 10
df_temp0 = copy.deepcopy(df0)
for i in range(shift_num):
    df0 = pd.concat([df0, df_temp0.groupby('route').shift(periods=i + 1).add_suffix(f'_shifted{i + 1}')], axis=1)

df0.dropna(inplace=True)

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)

data_trans = copy.deepcopy(df0)
data_trans.dropna(inplace=True)
df2 = copy.deepcopy(data_trans)
df2.reset_index(inplace=True, drop=True)
# Define the column you want to predict and the columns you want to use as features
col_predict = 'baggage'
features = list(df2.columns[:-1])

# Split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df2[features], df2[col_predict], test_size=0.1, shuffle=False)
# x_train = x_train0[x_train0['year'] >= 2022]
# y_train = y_train0[x_train0['year'] >= 2022]

model = keras.models.load_model('my_model_baggage.h5')

y_pred_train = model.predict(x_train).reshape(-1, 1)
y_actual = y_train.values.reshape(-1, 1)
df_result_train = pd.DataFrame({'pred_train': y_pred_train.reshape(-1),
                          'actual_train': y_actual.reshape(-1)})
df_result_train['error_train'] = df_result_train['actual_train'] - df_result_train['pred_train']
df_test = pd.concat([x_train.iloc[:,:8],y_train,df_result_train], axis=1)
df_test2 = df_test[df_test['error_train']>=300]
df_test2['route'] = le_route.inverse_transform(df_test2['route'].values)

model2 = model_sklearn()
model2.fit(x_train[df_result_train['error_train']>=300],y_train[df_result_train['error_train']>=300])

y_pred_train2 = model2.predict(x_train[df_result_train['error_train']>=300])
mae_on_train2 = np.mean(np.abs(y_train[df_result_train['error_train']>=300]-y_pred_train2))

y_pred_test = model.predict(x_test)
y_pred_test2 = model2.predict(x_test)

x_train_selection = np.concatenate((x_train, model.predict(x_train).reshape(-1,1), model2.predict(x_train).reshape(-1,1)), axis=1)
x_test_selection = np.concatenate((x_test, model.predict(x_test).reshape(-1,1), model2.predict(x_test).reshape(-1,1)), axis=1)


input_shape = x_train_selection.shape[1]
input_layer = keras.Input(shape=input_shape)
x = layers.Dense(100, activation="relu")(input_layer)
x = layers.Dense(50, activation="relu")(x)
# x = layers.Dense(50, activation="relu")(x)
# x = layers.Dense(50, activation="relu")(x)
# x = layers.Dense(50, activation="relu")(x)
# x = layers.Dense(50, activation="relu")(x)
# x = layers.Dense(50, activation="relu")(x)
x = layers.Dense(5, activation="relu")(x)
# x = layers.Flatten()(x)
output_layer = layers.Dense(1)(x)
model_decider = keras.Model(inputs=input_layer, outputs=output_layer)
model_decider.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

history = model_decider.fit(x_train_selection, y_train,
                    validation_data=(x_test_selection, y_test), callbacks=es, epochs=10000, batch_size=100)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred = model_decider.predict(x_test_selection).reshape(-1, 1)
y_actual = y_test.values.reshape(-1, 1)

df_result = pd.DataFrame({'pred': y_pred.reshape(-1),
                          'actual': y_actual.reshape(-1)})
df_result['error'] = df_result['actual'] - df_result['pred']

for i in range(0, 1000, 100):
    print(f'Size of the errors between {i}kg to {i + 100}kg',
          len(abs(df_result['error'])[((abs(df_result['error']) >= i) & (abs(df_result['error']) < i + 100))]) / len(
              df_result['error']))
print(np.mean(np.abs(df_result['error'])))
# boosted_ann.save('model1_baggage_boosting.h5')
