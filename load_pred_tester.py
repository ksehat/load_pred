import copy
import numpy as np
import requests
from functions import api_token_handler
from load_pred import mean_load_pred2
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


def model_tracker(model, x, y):
    # Split the data into training and validation sets
    kf = KFold(n_splits=50, shuffle=False)

    # Initialize empty lists to track the training and validation losses
    train_losses = []
    val_losses = []
    models = []
    x, y = np.array(x), np.array(y)
    # Loop over the K folds and track the training and validation losses
    for train_index, val_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)
        train_loss = mae(y_train, y_train_pred)
        val_loss = mae(y_val, y_val_pred)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        models.append(model)

    index_best_model = val_losses.index(min(val_losses[1:]))


    # Plot the training and validation losses over K folds
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()

    return models[index_best_model]

# Load your data into a DataFrame
data = pd.read_csv('df.csv')
data = data.filter(['is_holiday', 'year', 'month', 'day', 'dayofweek', 'PaxWeight'])
le = LabelEncoder()
data['year'] = le.fit_transform(data['year'])

data_trans = copy.deepcopy(data)
pax_weight_transformer = StandardScaler()
data_trans['PaxWeight'][:-1] = \
    pax_weight_transformer.fit_transform(data['PaxWeight'][:-1].values.reshape(-1, 1)).reshape(1, -1)[0]

# Adding new features related to last days
data_trans_shifted = pd.concat([data_trans[data_trans.columns[:-1]], data_trans.shift(1).add_suffix('_shifted1')], axis=1)
data_trans_shifted = pd.concat([data_trans_shifted, data_trans.shift(2).add_suffix('_shifted2')], axis=1)
data_trans_shifted = pd.concat([data_trans_shifted, data_trans['PaxWeight']], axis=1)
data_trans_shifted.dropna(inplace=True)
data_trans_shifted.reset_index(inplace=True, drop=True)




# Define the columns you want to use as features
features = list(data_trans_shifted.columns[:-1])

# Create a PolynomialFeatures object with the desired degree
poly = PolynomialFeatures(degree=5)

# Fit and transform your data
poly_features = poly.fit_transform(data_trans_shifted[features])

# Create a list of column names for the polynomial features DataFrame
poly_columns = []
for i in range(poly_features.shape[1]):
    poly_columns.append(f'poly_{i}')

# Create a new DataFrame with the polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly_columns)

# Concatenate the original DataFrame with the polynomial features DataFrame
data_temp = pd.concat([data_trans_shifted[features], poly_df], axis=1)
df1 = pd.concat([data_temp, data_trans_shifted['PaxWeight']], axis=1)

# # Create a KBinsDiscretizer object with the desired number of bins and strategy
# discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
#
# # Fit and transform your data
# col_to_bin = list(df1.columns[:-1])
# binned_data = pd.DataFrame(discretizer.fit_transform(df1[col_to_bin]))
# df = pd.concat([binned_data, data_trans['PaxWeight']], axis=1)


# Define the column you want to predict and the columns you want to use as features
col_predict = 'PaxWeight'
features = list(df1.columns[:-1])

# Split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df1[features], df1[col_predict], test_size=0.1, shuffle=False)

# Create and train a GradientBoostingRegressor
# model = GradientBoostingRegressor(n_estimators=500, max_depth=4, random_state=3, n_iter_no_change=100)
model = MLPRegressor(hidden_layer_sizes=(100,40,20,10,5,2), learning_rate='adaptive', early_stopping=1, learning_rate_init=0.1, max_iter=1000, verbose=1, shuffle=0, random_state=3, n_iter_no_change=100, tol=1e-5)
# model = AdaBoostRegressor(n_estimators=2000, random_state=3, learning_rate=0.01)
# model = LinearRegression()
# model = HuberRegressor(max_iter=50)
# model = TheilSenRegressor(max_iter=50)
# gbm.fit(x_train, y_train)

# selector = SelectFromModel(model, prefit=False).fit(x_train,y_train)
#
# x_train_selected = selector.transform(x_train)
# x_test_selected = selector.transform(x_test)
#
# # model.fit(x_train_selected, y_train)
#
model = model_tracker(model, x_train[:-1], y_train[:-1])
#
# # Make a prediction for the last row of data
# y_pred_untrans = model.predict(x_test_selected)
#
# # Inverse transformation of the output
# y_pred = pax_weight_transformer.inverse_transform(y_pred_untrans.reshape(-1, 1))
# y_actual = pax_weight_transformer.inverse_transform(y_test.values.reshape(-1, 1))

# model.fit(x_train, y_train)
y_pred = pax_weight_transformer.inverse_transform(model.predict(x_test).reshape(1, -1))
y_actual = pax_weight_transformer.inverse_transform(y_test.values.reshape(1, -1))

df_result = pd.DataFrame({'pred': y_pred.reshape(-1),
                          'actual': y_actual.reshape(-1)})
df_result['error'] = df_result['actual'] - df_result['pred']
print(np.mean(np.abs(df_result['error'])))
