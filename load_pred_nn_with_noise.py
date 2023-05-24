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
from create_model import manual_model
import pickle

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load your data into a DataFrame
df0 = pd.read_excel('data/df.xlsx')

df0['year'] = np.array(pd.DatetimeIndex(df0['Departure']).year)
df0['month'] = np.array(pd.DatetimeIndex(df0['Departure']).month)
df0['day'] = np.array(pd.DatetimeIndex(df0['Departure']).day)
df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['Departure']).dayofweek)
df0['hour'] = np.array(pd.DatetimeIndex(df0['Departure']).hour)

le_route = LabelEncoder()
df0['FlightRoute'] = le_route.fit_transform(df0['FlightRoute'])

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le_route, f)

df0.drop(['Departure', 'GregorianDate'], inplace=True, axis=1)


shift_num = 10
df_temp0 = copy.deepcopy(df0)
for i in range(shift_num):
    df0 = pd.concat([df0, df_temp0.groupby('FlightRoute').shift(periods=i+1).add_suffix(f'_shifted{i+1}')], axis=1)

df0.dropna(inplace=True)

filtered_columns_list = ['year', 'month', 'day', 'dayofweek', 'hour', 'FlightRoute', 'is_holiday']
all_org_rows_list = filtered_columns_list + ['PaxWeight']
for i in range(shift_num):
    filtered_columns_list_temp = [x+f'_shifted{i+1}' for x in all_org_rows_list]
    for x in filtered_columns_list_temp:
        filtered_columns_list.append(x)
filtered_columns_list.append('PaxWeight')

df1 = df0.filter(filtered_columns_list)

data_trans = copy.deepcopy(df1)

data_trans.dropna(inplace=True)

df2 = copy.deepcopy(data_trans)


# Define the column you want to predict and the columns you want to use as features
col_predict = 'PaxWeight'
features = list(df2.columns[:-1])

# Split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df2[features], df2[col_predict], test_size=0.1, shuffle=False)

# Adding noise
noise = np.random.normal(loc=0, scale=1, size=len(y_train))
y_train += noise

model = manual_model(x_train.values)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es = EarlyStopping(monitor='val_loss', mode='min', patience=200, restore_best_weights=True)
history = model.fit(x_train.values.reshape((x_train.values.shape[0], x_train.values.shape[1], 1)), y_train,
                    validation_data=(x_test, y_test), callbacks=es, epochs=10000, batch_size=100)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# selector = SelectFromModel(model, prefit=False).fit(x_train,y_train)
#
# x_train_selected = selector.transform(x_train)
# x_test_selected = selector.transform(x_test)
#
# # model.fit(x_train_selected, y_train)
#
# model = model_tracker(model, x_train[:-1], y_train[:-1])
#
# # Make a prediction for the last row of data
# y_pred_untrans = model.predict(x_test_selected)
#
# # Inverse transformation of the output
# y_pred = pax_weight_transformer.inverse_transform(y_pred_untrans.reshape(-1, 1))
# y_actual = pax_weight_transformer.inverse_transform(y_test.values.reshape(-1, 1))

# model.fit(x_train, y_train)
y_pred = model.predict(x_test).reshape(1, -1)
y_actual = y_test.values.reshape(1, -1)

df_result = pd.DataFrame({'pred': y_pred.reshape(-1),
                          'actual': y_actual.reshape(-1)})
df_result['error'] = df_result['actual'] - df_result['pred']
for i in range(15):
    print(f'Size of the errors between {i}kg to {i + 1}kg',
          len(abs(df_result['error'])[((abs(df_result['error']) >= i) & (abs(df_result['error']) < i + 1))]) / len(
              df_result['error']))
print(np.mean(np.abs(df_result['error'])))
model.save('my_model.h5')
