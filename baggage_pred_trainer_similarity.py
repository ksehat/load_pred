import copy
import pickle
import json
import numpy as np
import requests
import pandas as pd
from collections import defaultdict
import joblib
from joblib import dump
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from create_model import manual_model_dense
from functions import api_token_handler
from early_stopping_multiple import EarlyStoppingMultiple
from save_training_weights import SaveWeights
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from baggage_pred_pretrained_model import apply_label_dict


# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_last_5(df, reverse_route, date):
    mask = (df['route'] == reverse_route) & (df['departure'] < date)
    return df.loc[mask, 'baggage'].tail(5).tolist()


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
    dump(label_dict, 'artifacts/baggage/baggage_deployed_models/label_dict.joblib')
    return encoded_column


# Load your data into a DataFrame
# token = api_token_handler()
# df0 = pd.DataFrame(
#     json.loads(requests.get(url='http://192.168.115.10:8081/api/FlightBaggageEstimate/GetAllPastFlightsBaggage',
#                             headers={'Authorization': f'Bearer {token}',
#                                      'Content-type': 'application/json',
#                                      }
#                             ).text)['getAllPastFlightsBaggageResponseItemViewModels']).sort_values(by='departure')
#
# df0.drop(['pkFlightInformation'], axis=1, inplace=True)
#
# df0['baggage'] = df0['baggage'].str.split('/', expand=True)[1]
# df0['baggage'] = df0['baggage'].str.split(' ', expand=True)[0]
# df0['baggage'] = df0['baggage'].astype(float)
#
# df0['year'] = np.array(pd.DatetimeIndex(df0['departure']).year)
# df0['month'] = np.array(pd.DatetimeIndex(df0['departure']).month)
# df0['day'] = np.array(pd.DatetimeIndex(df0['departure']).day)
# df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['departure']).dayofweek)
# df0['hour'] = np.array(pd.DatetimeIndex(df0['departure']).hour)
# df0['quarter'] = np.array(pd.DatetimeIndex(df0['departure']).quarter)
#
# df0['departure'] = pd.to_datetime(df0['departure'])
# df0.sort_values(by='departure', inplace=True)
# df0.reset_index(drop=True, inplace=True)
#
# # Apply the function to the route column and assign it to a new column
# df0['route'] = custom_label_encode(df0['route'])
# # create a function to get the last 5 values of the baggage column for the reverse route
# last_5_values = df0.apply(lambda x: get_last_5(df0, -x['route'], x['departure']), axis=1)
# for k1 in range(5):
#     df0[f'reverse_baggage_{k1 + 1}'] = last_5_values.apply(lambda x: x[k1] if len(x) > k1 else None)
#
# df0.drop(['departure', 'paxWeight'], inplace=True, axis=1)
#
# col = df0.pop('baggage')
# df0.insert(len(df0.columns), 'baggage', col)
#
# shift_num = 15
# df_temp0 = copy.deepcopy(df0)
# for i in range(shift_num):
#     df0 = pd.concat([df0, df_temp0.groupby('route').shift(periods=i + 1).add_suffix(f'_shifted{i + 1}')], axis=1)
#
# col = df0.pop('baggage')
# df0.insert(len(df0.columns), 'baggage', col)
#
# ### IMPORTANT: DO NOT REMOVE THIS SECTION
# # region new routes prediction which has many None values
# mask = df0.iloc[5000:].isnull().any(axis=1)
# # Use boolean indexing to filter the rows
# df0_for_new_routes = df0.iloc[5000:][mask]
#
# from sklearn.ensemble import HistGradientBoostingRegressor as hgbr
# model_for_new_routes = hgbr(max_depth=6, random_state=40)
# x_train_new_routes = df0_for_new_routes.iloc[:-1,:-1]
# x_test_new_routes = df0_for_new_routes.iloc[-1:,:-1]
# y_train_new_routes = df0_for_new_routes.iloc[:-1,-1:]
# y_test_new_routes = df0_for_new_routes.iloc[-1:,-1:]
#
# model_for_new_routes.fit(x_train_new_routes, y_train_new_routes)
# y_pred_new_routes = model_for_new_routes.predict(x_test_new_routes)
#
# from sklearn.metrics import mean_absolute_error as mae
# print(mae(y_test_new_routes,y_pred_new_routes))
# filename = 'artifacts/baggage/baggage_deployed_models/hgbr.pkl'
# pickle.dump(model_for_new_routes, open(filename, 'wb'))
# # endregion
#
# df0.dropna(inplace=True)
#
# df1 = copy.deepcopy(df0)
# df1.reset_index(drop=True, inplace=True)
# n = 3  # number of similar rows to find
# similarity_columns = list(df1.columns)[:-1]
# df1_np = df1[similarity_columns].to_numpy()
#
# # Find the n most similar rows for each row
# most_similar_rows = []
# for i in range(n + 1, len(df1)):
#     # Build a KDTree with the rows before the current row
#     if i >= 100:
#         tree = KDTree(df1_np[i - 100:i + 1])
#     else:
#         tree = KDTree(df1_np[:i + 1])
#
#     # Find the n most similar rows
#     dist, ind = tree.query(df1_np[i:i + 1], k=n + 1)
#     most_similar_indices = ind[0][:n+1] + i - 100 if i >=100 else ind[0][:n+1]
#     most_similar_rows.append(df1.iloc[most_similar_indices].stack().to_frame().reset_index(drop=True).T)
#
# # Create a new dataframe with the most similar rows as new columns
# arr1 = np.concatenate((np.array(most_similar_rows).squeeze(), df1.to_numpy()[n + 1:, -1].reshape(-1, 1)),
#     axis=1)
# arr3 = np.delete(arr1, 272, axis=1)
#
# ss = Normalizer()
# arr1 = ss.fit_transform(arr3[:, :-1])
#
# filename = 'artifacts/baggage/baggage_deployed_models/scaler.pkl'
# pickle.dump(ss, open(filename, 'wb'))
#
# arr2 = np.concatenate((arr1, arr3[:, -1].reshape(-1, 1)), axis=1)
#
# x_train = arr2[:23000, :-1]
# x_test = arr2[23000:23700, :-1]
# y_train = arr2[:23000, -1]
# y_test = arr2[23000:23700, -1]
# #
# np.savetxt('artifacts/baggage/data/x_train_similarity.csv', x_train, delimiter=',')
# np.savetxt('artifacts/baggage/data/x_test_similarity.csv', x_test, delimiter=',')
# np.savetxt('artifacts/baggage/data/y_train_similarity.csv', y_train, delimiter=',')
# np.savetxt('artifacts/baggage/data/y_test_similarity.csv', y_test, delimiter=',')
# =====================================================================================================
x_train = np.loadtxt('artifacts/baggage/data/x_train_similarity.csv', delimiter=',')
x_test = np.loadtxt('artifacts/baggage/data/x_test_similarity.csv', delimiter=',')
y_train = np.loadtxt('artifacts/baggage/data/y_train_similarity.csv', delimiter=',')
y_test = np.loadtxt('artifacts/baggage/data/y_test_similarity.csv', delimiter=',')
# =====================================================================================================
# model1 = manual_model_dense(x_train)
# model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#                loss=tf.keras.losses.Huber(), metrics='mae')
#
# history = model1.fit(x_train, y_train,
#                      validation_data=(x_test, y_test), callbacks=[
#         SaveWeights('C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage/baggage_similarity_training_weights/model1/')],
#                      epochs=10000,
#                      batch_size=50)
# model1.save('artifacts/baggage/baggage_deployed_models/baggage_model1.h5')
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

model1 = keras.models.load_model('artifacts/baggage/baggage_deployed_models/baggage_model1.h5')
model1.load_weights(
    'C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage/baggage_similarity_training_weights/model1/weights_epoch312.h5')

y_pred_train1 = model1.predict(x_train)
y_pred_test1 = model1.predict(x_test)
x_train2 = x_train[(abs(y_pred_train1.reshape(-1) - y_train) >= 700)]
y_train2 = y_train[(abs(y_pred_train1.reshape(-1) - y_train) >= 700)]
# =====================================================================================================
# model2 = GradientBoostingRegressor(max_depth=5)
# model2.fit(x_train2, y_train2)
#
# filename = 'artifacts/baggage/baggage_deployed_models/baggage_model2.sav'
# pickle.dump(model2, open(filename, 'wb'))

model2 = joblib.load('artifacts/baggage/baggage_deployed_models/baggage_model2.sav')

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
               loss=tf.keras.losses.Huber(), metrics='mae')

history = model3.fit(x_train3, y_train,
                     validation_data=(x_test3, y_test), callbacks=[
        SaveWeights('C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage/baggage_similarity_training_weights/model3/')],
                     epochs=10000,
                     batch_size=50)
model3.save('artifacts/baggage/baggage_deployed_models/baggage_model3.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model3 = keras.models.load_model('artifacts/baggage/baggage_deployed_models/baggage_model3.h5')
model3.load_weights(
    'C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage/baggage_similarity_training_weights\model3/weights_epoch762.h5')
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

# from sklearn.metrics import mean_absolute_error as mae
#
# x_val = df2[:21944, :-1]
# y_true = df2[:21944, -1]
# y_pred1 = model1.predict(x_val)
# y_pred2 = model2.predict(x_val)
# x_val_final = np.concatenate((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)), axis=1)
# y_pred_final = model3.predict(x_val_final)
#
# error = mae(y_true, y_pred_final)
# print(error)
