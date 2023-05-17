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
from create_model import create_model
from create_model import create_manual_model

# Load your data into a DataFrame
df0 = pd.read_excel('data/df.xlsx')

df0['year'] = np.array(pd.DatetimeIndex(df0['Departure']).year)
df0['month'] = np.array(pd.DatetimeIndex(df0['Departure']).month)
df0['day'] = np.array(pd.DatetimeIndex(df0['Departure']).day)
df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['Departure']).dayofweek)
df0['hour'] = np.array(pd.DatetimeIndex(df0['Departure']).hour)

le_route = LabelEncoder()
df0['FlightRoute'] = le_route.fit_transform(df0['FlightRoute'])
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
# df1 = df1.iloc[100:, :]

# le = LabelEncoder()
# df1['year'] = le.fit_transform(df1['year'])

data_trans = copy.deepcopy(df1)
# pax_weight_transformer = StandardScaler()
# ss_transformer = StandardScaler()
#
# data_trans['PaxWeight'] = pd.DataFrame(
#     pax_weight_transformer.fit_transform(data_trans['PaxWeight'].values.reshape(-1, 1)), index=data_trans.index)
# data_trans.iloc[:, :-1] = pd.DataFrame(ss_transformer.fit_transform(data_trans.iloc[:, :-1]),
#                                        columns=data_trans.columns[:-1])

data_trans.dropna(inplace=True)

# Adding new features related to last days
# data_trans_shifted = pd.concat([data_trans[data_trans.columns[:-1]], data_trans.shift(1).add_suffix('_shifted1')], axis=1)
# data_trans_shifted = pd.concat([data_trans_shifted, data_trans.shift(2).add_suffix('_shifted2')], axis=1)
# data_trans_shifted = pd.concat([data_trans_shifted, data_trans['PaxWeight']], axis=1)
# data_trans_shifted.dropna(inplace=True)
# data_trans_shifted.reset_index(inplace=True, drop=True)

df2 = copy.deepcopy(data_trans)

# # Define the columns you want to use as features
# features = list(data_trans_shifted.columns[:-1])
#
# # Create a PolynomialFeatures object with the desired degree
# poly = PolynomialFeatures(degree=5)
#
# # Fit and transform your data
# poly_features = poly.fit_transform(data_trans_shifted[features])
#
# # Create a list of column names for the polynomial features DataFrame
# poly_columns = []
# for i in range(poly_features.shape[1]):
#     poly_columns.append(f'poly_{i}')
#
# # Create a new DataFrame with the polynomial features
# poly_df = pd.DataFrame(poly_features, columns=poly_columns)
#
# # Concatenate the original DataFrame with the polynomial features DataFrame
# data_temp = pd.concat([data_trans_shifted[features], poly_df], axis=1)
# df1 = pd.concat([data_temp, data_trans_shifted['PaxWeight']], axis=1)

# # Create a KBinsDiscretizer object with the desired number of bins and strategy
# discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
#
# # Fit and transform your data
# col_to_bin = list(df1.columns[:-1])
# binned_data = pd.DataFrame(discretizer.fit_transform(df1[col_to_bin]))
# df = pd.concat([binned_data, data_trans['PaxWeight']], axis=1)


# Define the column you want to predict and the columns you want to use as features
col_predict = 'PaxWeight'
features = list(df2.columns[:-1])

# Split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df2[features], df2[col_predict], test_size=0.1, shuffle=False)

# Create and train a GradientBoostingRegressor
# model = GradientBoostingRegressor(n_estimators=500, max_depth=4, random_state=3, n_iter_no_change=100)
# model = MLPRegressor(hidden_layer_sizes=(100,40,20,10,5,2), learning_rate='adaptive', early_stopping=1, learning_rate_init=0.1, max_iter=1000, verbose=1, shuffle=0, random_state=3, n_iter_no_change=100, tol=1e-5)
# model = AdaBoostRegressor(n_estimators=2000, random_state=3, learning_rate=0.01)
# model = LinearRegression()
# model = HuberRegressor(max_iter=50)

# model = TheilSenRegressor(max_iter=50)

# model = create_nlp_model(x_train.shape[1], [ 500, 100, 50, 20,5])
# model = create_model(x_train.values, ['Conv1D', 'Conv1D', 'Conv1D', 'Dense', 'Dense'], [50, 20, 10, 10, 5])
model = create_manual_model(x_train.values)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.MeanAbsoluteError(), metrics='mae')

es = EarlyStopping(monitor='loss', mode='min', patience=200, restore_best_weights=True)
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
