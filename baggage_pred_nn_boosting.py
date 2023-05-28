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
from create_model_baggage import manual_model
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load your data into a DataFrame
df0 = pd.read_excel('data/df_baggage.xlsx')

df0.rename(columns={'PaxWeigth': 'PaxWeight'}, inplace=True)

df0['year'] = np.array(pd.DatetimeIndex(df0['Departure']).year)
df0['month'] = np.array(pd.DatetimeIndex(df0['Departure']).month)
df0['day'] = np.array(pd.DatetimeIndex(df0['Departure']).day)
df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['Departure']).dayofweek)
df0['hour'] = np.array(pd.DatetimeIndex(df0['Departure']).hour)

le_route = LabelEncoder()
df0['FlightRoute'] = le_route.fit_transform(df0['FlightRoute'])

with open('label_encoder_baggage_boosting.pkl', 'wb') as f:
    pickle.dump(le_route, f)

df0.drop(['Departure', 'pkFlightInformation'], inplace=True, axis=1)

shift_num = 10
df_temp0 = copy.deepcopy(df0)
for i in range(shift_num):
    df0 = pd.concat([df0, df_temp0.groupby('FlightRoute').shift(periods=i + 1).add_suffix(f'_shifted{i + 1}')], axis=1)

df0.dropna(inplace=True)

filtered_columns_list = ['year', 'month', 'day', 'dayofweek', 'hour', 'FlightRoute', 'is_holiday', 'Seats', 'PaxWeight']
all_org_rows_list = filtered_columns_list + ['BaggageWeight']
for i in range(shift_num):
    filtered_columns_list_temp = [x + f'_shifted{i + 1}' for x in all_org_rows_list]
    for x in filtered_columns_list_temp:
        filtered_columns_list.append(x)
filtered_columns_list.append('BaggageWeight')

df1 = df0.filter(filtered_columns_list)
data_trans = copy.deepcopy(df1)
data_trans.dropna(inplace=True)
df2 = copy.deepcopy(data_trans)

# Define the column you want to predict and the columns you want to use as features
col_predict = 'BaggageWeight'
features = list(df2.columns[:-1])

# Split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df2[features], df2[col_predict], test_size=0.1, shuffle=False)


es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)

ann_estimator = KerasRegressor(build_fn=lambda: manual_model(x_train.values), epochs=10000, batch_size=300, verbose=1,
                               callbacks=[es],
                               loss=tf.keras.losses.MeanAbsoluteError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
boosted_ann = AdaBoostRegressor(base_estimator=ann_estimator, n_estimators=5)
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
