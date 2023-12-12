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
from create_model import manual_model_dense_baggage_pax
from functions import api_token_handler
from early_stopping_multiple import EarlyStoppingMultiple
from save_training_weights import SaveWeights
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import HistGradientBoostingRegressor as hgbr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics.pairwise import cosine_similarity
from baggage_pred_pretrained_model import apply_label_dict


# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def new_routes_prediction(pred_col):
    mask = df0.iloc[5000:].isnull().any(axis=1)
    # Use boolean indexing to filter the rows
    df0_for_new_routes = df0.iloc[5000:][mask]

    model_for_new_routes = hgbr(max_depth=6, random_state=40)
    x_cols = [col for col in df0_for_new_routes.columns if col not in ['baggage', 'paxWeight']]
    x_train_new_routes = df0_for_new_routes.iloc[:-100][x_cols]
    x_test_new_routes = df0_for_new_routes.iloc[-100:][x_cols]
    y_train_new_routes = df0_for_new_routes.iloc[:-100][pred_col]
    y_test_new_routes = df0_for_new_routes.iloc[-100:][pred_col]

    model_for_new_routes.fit(x_train_new_routes, y_train_new_routes)
    y_pred_new_routes = model_for_new_routes.predict(x_test_new_routes)

    print(mae(y_test_new_routes, y_pred_new_routes))
    filename = f'artifacts/baggage_pax/baggage_pax_deployed_models/hgbr_{pred_col}.pkl'
    pickle.dump(model_for_new_routes, open(filename, 'wb'))


def get_last_5_baggage(df, reverse_route, date):
    mask = (df['route'] == reverse_route) & (df['departure'] < date)
    return df.loc[mask, 'baggage'].tail(5).tolist()


def get_last_5_pax(df, reverse_route, date):
    mask = (df['route'] == reverse_route) & (df['departure'] < date)
    return df.loc[mask, 'paxWeight'].tail(5).tolist()


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
    dump(label_dict, 'artifacts/baggage_pax/baggage_pax_deployed_models/label_dict.joblib')
    return encoded_column


# Load your data into a DataFrame
token = api_token_handler()
df0 = pd.DataFrame(
    json.loads(requests.get(url='http://192.168.115.10:8081/api/FlightBaggageEstimate/GetAllPastFlightsBaggage',
                            headers={'Authorization': f'Bearer {token}',
                                     'Content-type': 'application/json',
                                     }
                            ).text)['getAllPastFlightsBaggageResponseItemViewModels']).sort_values(by='departure')

df0.drop(['pkFlightInformation'], axis=1, inplace=True)

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
# create a function to get the last 5 values of the baggage column for the reverse route
last_5_values_baggage = df0.apply(lambda x: get_last_5_baggage(df0, -x['route'], x['departure']), axis=1)
for k1 in range(5):
    df0[f'reverse_baggage_{k1 + 1}'] = last_5_values_baggage.apply(lambda x: x[k1] if len(x) > k1 else None)
# create a function to get the last 5 values of the paxWeight column for the reverse route
last_5_values_pax = df0.apply(lambda x: get_last_5_pax(df0, -x['route'], x['departure']), axis=1)
for k1 in range(5):
    df0[f'reverse_pax_{k1 + 1}'] = last_5_values_pax.apply(lambda x: x[k1] if len(x) > k1 else None)

df0.drop(['departure'], inplace=True, axis=1)

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)
col = df0.pop('paxWeight')
df0.insert(len(df0.columns), 'paxWeight', col)

shift_num = 15
df_temp0 = copy.deepcopy(df0)
for i in range(shift_num):
    df0 = pd.concat([df0, df_temp0.groupby('route').shift(periods=i + 1).add_suffix(f'_shifted{i + 1}')], axis=1)

col = df0.pop('baggage')
df0.insert(len(df0.columns), 'baggage', col)
col = df0.pop('paxWeight')
df0.insert(len(df0.columns), 'paxWeight', col)

### IMPORTANT: DO NOT REMOVE THIS SECTION
new_routes_prediction('baggage')
new_routes_prediction('paxWeight')

df0.dropna(inplace=True)
df1 = copy.deepcopy(df0)
df1.reset_index(drop=True, inplace=True)

n = 3  # number of similar rows to find
similarity_columns = list(df1.columns)[:-2]
df1_np = df1[similarity_columns].to_numpy()

# Find the n most similar rows for each row
most_similar_rows = []
baggage_indice_in_df1 = len(df1.columns)-2
pax_indice_in_df1 = len(df1.columns)-1
for i in range(n + 1, len(df1)):
    # Build a KDTree with the rows before the current row
    if i >= 100:
        tree = KDTree(df1_np[i - 100:i + 1])
    else:
        tree = KDTree(df1_np[:i + 1])

    # Find the n most similar rows
    dist, ind = tree.query(df1_np[i:i + 1], k=n + 1)
    most_similar_indices = ind[0][:n+1] + i - 100 if i >=100 else ind[0][:n+1]
    most_similar_rows.append(df1.iloc[most_similar_indices].stack().to_frame().reset_index(drop=True).T)

# Create a new dataframe with the most similar rows as new columns
arr1 = np.concatenate((np.array(most_similar_rows).squeeze(), df1.to_numpy()[n + 1:, -2:].reshape(-1, 2)), axis=1)
#arr1 has baggage and paxWeight both in the last two columns and also between the columns in the middle
# Because of this we delete the baggage_indice_in_df1 and pax_indice_in_df1 columns in the next two lines
arr2 = np.delete(arr1, baggage_indice_in_df1, axis=1)
arr3 = np.delete(arr2, pax_indice_in_df1, axis=1) # array 3 has all columns and the baggage and paxWeight are the last columns

ss = Normalizer()
arr4 = ss.fit_transform(arr3[:, :-2])

filename = 'artifacts/baggage_pax/baggage_pax_deployed_models/scaler.pkl'
pickle.dump(ss, open(filename, 'wb'))

arr5 = np.concatenate((arr4, arr3[:, -2:].reshape(-1, 2)), axis=1)

x_train = arr5[:-60, :-2]
x_test = arr5[-60:, :-2]
y_train = arr5[:-60, -2:]
y_test = arr5[-60:, -2:]

np.savetxt('artifacts/baggage_pax/data/x_train_similarity.csv', x_train, delimiter=',')
np.savetxt('artifacts/baggage_pax/data/x_test_similarity.csv', x_test, delimiter=',')
np.savetxt('artifacts/baggage_pax/data/y_train_similarity.csv', y_train, delimiter=',')
np.savetxt('artifacts/baggage_pax/data/y_test_similarity.csv', y_test, delimiter=',')
# =====================================================================================================
x_train = np.loadtxt('artifacts/baggage_pax/data/x_train_similarity.csv', delimiter=',')
x_test = np.loadtxt('artifacts/baggage_pax/data/x_test_similarity.csv', delimiter=',')
y_train = np.loadtxt('artifacts/baggage_pax/data/y_train_similarity.csv', delimiter=',')
y_test = np.loadtxt('artifacts/baggage_pax/data/y_test_similarity.csv', delimiter=',')
# =====================================================================================================
model1 = manual_model_dense_baggage_pax(x_train)
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
               loss=tf.keras.losses.Huber(), metrics='mae')

history = model1.fit(x_train, y_train,
                     validation_data=(x_test, y_test), callbacks=[
        SaveWeights('C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage_pax/baggage_pax_similarity_training_weights/model1/')],
                     epochs=10000,
                     batch_size=50)
model1.save('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model1.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model1 = keras.models.load_model('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model1.h5')
model1.load_weights(
    'C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage_pax/baggage_pax_similarity_training_weights/model1/weights_epoch128.h5')

y_pred_train1 = model1.predict(x_train)
y_pred_test1 = model1.predict(x_test)
x_train2 = x_train[(abs(y_pred_train1[:, 0].reshape(-1) - y_train[:, 0]) >= 800)]
y_train2 = y_train[(abs(y_pred_train1[:, 0].reshape(-1) - y_train[:, 0]) >= 800)]
# # =====================================================================================================
model2_baggage = GradientBoostingRegressor(max_depth=5)
model2_baggage.fit(x_train2, y_train2[:, 0])
model2_pax = GradientBoostingRegressor(max_depth=5)
model2_pax.fit(x_train2, y_train2[:, 1])

filename = 'artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model2.sav'
pickle.dump(model2_baggage, open(filename, 'wb'))
filename = 'artifacts/baggage_pax/baggage_pax_deployed_models/pax_model2.sav'
pickle.dump(model2_pax, open(filename, 'wb'))

model2_baggage = joblib.load('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model2.sav')
model2_pax = joblib.load('artifacts/baggage_pax/baggage_pax_deployed_models/pax_model2.sav')

y_pred_train2_baggage = model2_baggage.predict(x_train)
y_pred_test2_baggage = model2_baggage.predict(x_test)
x_train3_baggage = np.concatenate((y_pred_train1[:, 0].reshape(-1, 1), y_pred_train2_baggage.reshape(-1, 1)), axis=1)
x_test3_baggage = np.concatenate((y_pred_test1[:, 0].reshape(-1, 1), y_pred_test2_baggage.reshape(-1, 1)), axis=1)

y_pred_train2_pax = model2_pax.predict(x_train)
y_pred_test2_pax = model2_pax.predict(x_test)
x_train3_pax = np.concatenate((y_pred_train1[:, 1].reshape(-1, 1), y_pred_train2_pax.reshape(-1, 1)), axis=1)
x_test3_pax = np.concatenate((y_pred_test1[:, 1].reshape(-1, 1), y_pred_test2_pax.reshape(-1, 1)), axis=1)

x_train3 = np.concatenate((x_train3_baggage, x_train3_pax), axis=1)
x_test3 = np.concatenate((x_test3_baggage, x_test3_pax), axis=1)
# =====================================================================================================
input_shape = x_train3.shape[1]
input_layer = keras.Input(shape=input_shape)
x = input_layer
x1 = layers.Dense(100, activation="relu")(x)
x2 = layers.Dense(40, activation="relu")(x1)
x3 = layers.Dense(20, activation="relu")(x2)
x4 = layers.Dense(5, activation="relu")(x3)
# x5 = layers.Dense(5, activation="relu")(x4)
# output_layer1 = layers.Dense(3)(x5)
output_layer = layers.Dense(2)(x4)
model3 = keras.Model(inputs=input_layer, outputs=output_layer)
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
               loss=tf.keras.losses.Huber(), metrics='mae')

history = model3.fit(x_train3, y_train,
                     validation_data=(x_test3, y_test), callbacks=[
        SaveWeights(
            'C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage_pax/baggage_pax_similarity_training_weights/model3/')],
                     epochs=10000,
                     batch_size=100)
model3.save('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model3.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model3 = keras.models.load_model('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model3.h5')
model3.load_weights(
    'C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage_pax/baggage_pax_similarity_training_weights/model3/weights_epoch38.h5')
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
