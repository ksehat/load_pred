import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import keras
from create_model import manual_model
import pickle
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import layers


def model2(x_train2):
    input_shape = [x_train2.shape[1], 1]
    input_layer = keras.Input(shape=input_shape)
    x = input_layer
    x1 = layers.Conv1D(50, kernel_size=20, activation="relu")(x)
    x1 = layers.Dropout(0.25)(x1)
    # x2 = layers.Conv1D(20, kernel_size=10, activation="relu")(x1)
    x3 = layers.Dense(50, activation="relu")(x1)
    x4 = layers.Dense(25, activation="relu")(x3)
    x4 = layers.Dropout(0.25)(x4)
    x5 = layers.Dense(5, activation="relu")(x4)
    x6 = layers.Flatten()(x5)
    output_layer = layers.Dense(1)(x6)
    model2 = keras.Model(inputs=input_layer, outputs=output_layer)

    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')
    return model2

def model3(x_train3):
    input_shape = [x_train3.shape[1], 1]
    input_layer = keras.Input(shape=input_shape)
    x = input_layer
    # x1 = layers.Conv1D(30, kernel_size=2, activation="relu")(x)
    # x1 = layers.Dropout(0.25)(x1)
    # x2 = layers.Conv1D(20, kernel_size=1, activation="relu")(x1)
    x3 = layers.Dense(50, activation="relu")(x)
    x4 = layers.Dense(25, activation="relu")(x3)
    x5 = layers.Dense(5, activation="relu")(x4)
    x6 = layers.Flatten()(x5)
    output_layer = layers.Dense(1)(x6)
    model3 = keras.Model(inputs=input_layer, outputs=output_layer)

    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')
    return model3
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load your data into a DataFrame
df0 = pd.read_excel('data/df_baggage.xlsx')
df0.rename(columns={'PaxWeigth': 'PaxWeight'}, inplace=True)
# df0.drop(['PaxWeigth'], axis=1, inplace=True)

df0['year'] = np.array(pd.DatetimeIndex(df0['Departure']).year)
df0['month'] = np.array(pd.DatetimeIndex(df0['Departure']).month)
df0['day'] = np.array(pd.DatetimeIndex(df0['Departure']).day)
df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['Departure']).dayofweek)
df0['hour'] = np.array(pd.DatetimeIndex(df0['Departure']).hour)

le_route = LabelEncoder()
df0['FlightRoute'] = le_route.fit_transform(df0['FlightRoute'])

with open('label_encoder_baggage.pkl', 'wb') as f:
    pickle.dump(le_route, f)

df0.drop(['Departure', 'pkFlightInformation'], inplace=True, axis=1)

shift_num = 20
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

model = keras.models.load_model('my_model_baggage.h5')
y_pred_train = model.predict(x_train)
error = abs(y_train - y_pred_train.ravel())
high_error = error >= 200

x_train2 = x_train[high_error]
y_train2 = y_train[high_error]

model2 = model2(x_train2.values)
es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)
history2 = model2.fit(x_train2.values.reshape((x_train2.values.shape[0], x_train2.values.shape[1], 1)), y_train2,
                      callbacks=es, epochs=10000, batch_size=100)

y_pred_train1 = model.predict(x_train)
y_pred_train2 = model2.predict(x_train)

x_train3 = np.concatenate([y_pred_train1.reshape(-1, 1), y_pred_train2.reshape(-1, 1)], axis=1)


model3 = model3(x_train3)
es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)
history3 = model3.fit(x_train3.reshape((x_train3.shape[0], x_train3.shape[1], 1)), y_train,
                      callbacks=es, epochs=10000, batch_size=20)

y_pred1 = model.predict(x_test)
y_pred2 = model2.predict(x_test)

x_test3 = np.concatenate([y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)], axis=1)

y_pred3 = model3.predict(x_test3)
y_actual = y_test.values.reshape(1, -1)

df_result = pd.DataFrame({'pred': y_pred3.reshape(-1),
                          'actual': y_actual.reshape(-1)})
df_result['error'] = df_result['actual'] - df_result['pred']
for i in range(0, 1000, 100):
    print(f'Size of the errors between {i}kg to {i + 100}kg',
          len(abs(df_result['error'])[((abs(df_result['error']) >= i) & (abs(df_result['error']) < i + 100))]) / len(
              df_result['error']))
print(np.mean(np.abs(df_result['error'])))
# model.save('my_model_baggage.h5')
