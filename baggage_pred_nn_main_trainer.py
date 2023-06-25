import copy
import pickle
import json
import numpy as np
import requests
import pandas as pd
from collections import defaultdict
from joblib import dump
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
from early_stopping_multiple import EarlyStoppingMultiple

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def custom_label_encode(column):
    label_dict = defaultdict(int)
    label = 1
    encoded_column = []
    for route in column:
        origin, destination = route.split('>')
        reverse_route = f'{destination}>{origin}'
        if route in label_dict:
            encoded_column.append(label_dict[route])
        elif reverse_route in label_dict:
            encoded_column.append(-label_dict[reverse_route])
        else:
            label_dict[route] = label
            encoded_column.append(label)
            label += 1
    # Save the label_dict to a file
    dump(label_dict, 'baggage_deployed_models/label_dict.joblib')
    return encoded_column

# Load your data into a DataFrame
token = api_token_handler()
df0 = pd.DataFrame(
    json.loads(requests.get(url='http://192.168.115.10:8081/api/FlightBaggageEstimate/GetAllPastFlightsBaggage',
                            headers={'Authorization': f'Bearer {token}',
                                     'Content-type': 'application/json',
                                     }
                            ).text)['getAllPastFlightsBaggageResponseItemViewModels']).sort_values(by='departure')

df0.drop('pkFlightInformation', axis=1, inplace=True)

df0['baggage'] = df0['baggage'].str.split('/', expand=True)[1]
df0['baggage'] = df0['baggage'].str.split(' ', expand=True)[0]
df0['baggage'] = df0['baggage'].astype(float)

df0['year'] = np.array(pd.DatetimeIndex(df0['departure']).year)
df0['month'] = np.array(pd.DatetimeIndex(df0['departure']).month)
df0['day'] = np.array(pd.DatetimeIndex(df0['departure']).day)
df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['departure']).dayofweek)
df0['hour'] = np.array(pd.DatetimeIndex(df0['departure']).hour)
df0['quarter'] = np.array(pd.DatetimeIndex(df0['departure']).quarter)

df0['departure'] = pd.to_datetime(df0['departure'])
df0.sort_values(by='departure', inplace=True)
df0.reset_index(drop=True, inplace=True)

# Apply the function to the route column and assign it to a new column
df0['route'] = custom_label_encode(df0['route'])

df0.drop(['departure', 'paxWeight'], inplace=True, axis=1)

# add Gaussian noise to x column
# noise = np.random.normal(0, 1, len(df0))  # change mean and std as needed
# df0["paxWeight"] = df0["paxWeight"] + noise

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)

shift_num = 15
df_temp0 = copy.deepcopy(df0)
for i in range(shift_num):
    df0 = pd.concat([df0, df_temp0.groupby('route').shift(periods=i + 1).add_suffix(f'_shifted{i + 1}')], axis=1)

df0.dropna(inplace=True)

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)

# pf = PolynomialFeatures(degree=2)
# df1 = pf.fit_transform(df0.iloc[:,:-1])
# df2 = np.concatenate((df1,df0.iloc[:,-1:].values), axis=1)
df2 = copy.deepcopy(np.array(df0))
x_train, x_test, y_train, y_test = train_test_split(df2[:, :-1], df2[:, -1], test_size=0.1, shuffle=False)
# =====================================================================================================
model1 = manual_model_dense(x_train)
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
               loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es1 = EarlyStoppingMultiple(monitor1='loss', monitor2='val_loss', patience=10, fav_loss=186, fav_val_loss=185)
history = model1.fit(x_train, y_train,
                     validation_data=(x_test, y_test), callbacks=es1, epochs=10000, batch_size=50)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred_train1 = model1.predict(x_train)
y_pred_test1 = model1.predict(x_test)
x_train2 = x_train[(abs(y_pred_train1.reshape(-1) - y_train) >= 300)]
y_train2 = y_train[(abs(y_pred_train1.reshape(-1) - y_train) >= 300)]
# =====================================================================================================
model2 = GradientBoostingRegressor(max_depth=5)
model2.fit(x_train2, y_train2)
y_pred_train2 = model2.predict(x_train)
y_pred_test2 = model2.predict(x_test)
x_train3 = np.concatenate((y_pred_train1.reshape(-1, 1), y_pred_train2.reshape(-1, 1)), axis=1)
x_test3 = np.concatenate((y_pred_test1.reshape(-1, 1), y_pred_test2.reshape(-1, 1)), axis=1)
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
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
               loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es3 = EarlyStoppingMultiple(monitor1='loss', monitor2='val_loss', patience=10, fav_loss=165, fav_val_loss=180)
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


model1.save('baggage_deployed_models/baggage_model1.h5')
model1.save_weights('baggage_deployed_models/baggage_model1_weights.h5')
filename = 'baggage_deployed_models/baggage_model2.sav'
pickle.dump(model2, open(filename, 'wb'))
model3.save('baggage_deployed_models/baggage_model3.h5')
model3.save_weights('baggage_deployed_models/baggage_model3_weights.h5')
