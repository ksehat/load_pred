import copy
import numpy as np
import pandas as pd
import pickle
import json
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
from create_model_baggage import manual_model
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
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

data_trans = copy.deepcopy(df0)
data_trans.dropna(inplace=True)
df2 = copy.deepcopy(data_trans)

# Define the column you want to predict and the columns you want to use as features
col_predict = 'baggage'
features = list(df2.columns[:-1])

# Split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df2[features], df2[col_predict], test_size=0.1, shuffle=False)
# x_train = x_train0[x_train0['year'] >= 2022]
# y_train = y_train0[x_train0['year'] >= 2022]


es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)

ann_estimator = KerasRegressor(build_fn=lambda: manual_model(x_train.values), epochs=10000, batch_size=100, verbose=1,
                               callbacks=[es],
                               loss=tf.keras.losses.MeanAbsoluteError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
boosted_ann = AdaBoostRegressor(base_estimator=ann_estimator, n_estimators=3)
boosted_ann.fit(x_train, y_train) # scale your training data


y_pred = boosted_ann.predict(x_test).reshape(-1, 1)
y_actual = y_test.values.reshape(-1, 1)


df_result = pd.DataFrame({'pred': y_pred.reshape(-1),
                          'actual': y_actual.reshape(-1)})
df_result['error'] = df_result['actual'] - df_result['pred']

for i in range(0, 1000, 100):
    print(f'Size of the errors between {i}kg to {i + 100}kg',
          len(abs(df_result['error'])[((abs(df_result['error']) >= i) & (abs(df_result['error']) < i + 100))]) / len(
              df_result['error']))
print(np.mean(np.abs(df_result['error'])))
boosted_ann.save('model1_baggage_boosting.h5')
