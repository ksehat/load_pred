import copy
import numpy as np
import requests
from functions import api_token_handler
from load_pred2 import mean_load_pred2
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima


# Load your data into a DataFrame
data = pd.read_csv('df.csv')
data = data.filter(['is_holiday', 'year', 'month', 'day', 'dayofweek', 'PaxWeight'])
le = LabelEncoder()
data['year'] = le.fit_transform(data['year'])

data_trans = copy.deepcopy(data)
pax_weight_transformer = StandardScaler()
data_trans['PaxWeight'][:-1] = \
    pax_weight_transformer.fit_transform(data['PaxWeight'][:-1].values.reshape(-1, 1)).reshape(1, -1)[0]
# Define the columns you want to use as features
features = ['is_holiday', 'year', 'month', 'day', 'dayofweek']

# Create a PolynomialFeatures object with the desired degree
poly = PolynomialFeatures(degree=5)

# Fit and transform your data
poly_features = poly.fit_transform(data[features])

# Create a list of column names for the polynomial features DataFrame
poly_columns = []
for i in range(poly_features.shape[1]):
    poly_columns.append(f'poly_{i}')

# Create a new DataFrame with the polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly_columns)

# Concatenate the original DataFrame with the polynomial features DataFrame
data_temp = pd.concat([data[features], poly_df], axis=1)
df1 = pd.concat([data_temp, data_trans['PaxWeight']], axis=1)

# Create a KBinsDiscretizer object with the desired number of bins and strategy
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

# Fit and transform your data
col_to_bin = list(df1.columns[:-1])
binned_data = pd.DataFrame(discretizer.fit_transform(df1[col_to_bin]))
df = pd.concat([binned_data, data_trans['PaxWeight']], axis=1)


# Define the column you want to predict and the columns you want to use as features
col_predict = 'PaxWeight'
features = list(df.columns[:-1])

# Split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df[features], df[col_predict], test_size=0.1, shuffle=False)


# fit ARIMA model using pmdarima's auto_arima function
arima_model = auto_arima(train_data, start_p=1, start_q=1,
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)
arima_model_fit = arima_model.fit(train_data)

# make ARIMA prediction for validation data
arima_prediction = arima_model_fit.predict(n_periods=len(validation_data))

# prepare data for XGBoost
train_X, train_y = ...
validation_X, validation_y = ...
test_X = ...

# fit XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(train_X, train_y)

# make XGBoost prediction for validation data
xgb_prediction = xgb_model.predict(validation_X)

# calculate weights based on performance on validation data
arima_weight = 1 / mean_absolute_error(validation_data, arima_prediction)
xgb_weight = 1 / mean_absolute_error(validation_data, xgb_prediction)
total_weight = arima_weight + xgb_weight

# make final predictions for test data
arima_prediction = arima_model_fit.forecast(steps=len(test_data))[0]
xgb_prediction = xgb_model.predict(test_X)
final_prediction = (arima_weight * arima_prediction + xgb_weight * xgb_prediction) / total_weight



# Create and train a GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=500, max_depth=5, random_state=3, n_iter_no_change=1000)
gbm.fit(x_train, y_train)

selector = SelectFromModel(gbm, prefit=True)

x_train_selected = selector.transform(x_train)
x_test_selected = selector.transform(x_test)

gbm.fit(x_train_selected, y_train)



# Make a prediction for the last row of data
y_pred_untrans = gbm.predict(x_test_selected)

# Inverse transformation of the output
y_pred = pax_weight_transformer.inverse_transform(y_pred_untrans.reshape(-1, 1))
y_actual = pax_weight_transformer.inverse_transform(y_test.values.reshape(-1, 1))

df_result = pd.DataFrame({'pred': y_pred.reshape(len(y_pred)),
                          'actual': y_actual.reshape(len(y_actual))})
df_result['error'] = df_result['actual'] - df_result['pred']
print(np.mean(np.abs(df_result['error'])))
