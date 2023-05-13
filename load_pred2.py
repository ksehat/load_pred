import copy
import datetime
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer, KBinsDiscretizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, TheilSenRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel
from functions import api_token_handler




def mean_load_pred2(input_data_list):
    for input_data_dict in input_data_list:
        try:
            df = pd.DataFrame(input_data_dict['GetAllFlightsWithSameConditionRobotResponseItemViewModels'])
            PkFlightInformation = input_data_dict['PkFlightInformation']
            df_len = len(df)
            df['year'] = np.array(pd.DatetimeIndex(df['GregorianDate']).year)
            df['month'] = np.array(pd.DatetimeIndex(df['GregorianDate']).month)
            df['day'] = np.array(pd.DatetimeIndex(df['GregorianDate']).day)
            df['dayofweek'] = np.array(pd.DatetimeIndex(df['GregorianDate']).dayofweek)

            le = LabelEncoder()
            df['year'] = le.fit_transform(df['year'])

            df_temp = df[df['PaxWeight'] != 0]
            df = pd.concat([df_temp, df[-1:]])

            last_date = df.loc[0, 'GregorianDate']
            date_index = [0]
            for i in range(len(df)):
                if df['GregorianDate'][i] != last_date:
                    last_date = df['GregorianDate'][i]
                    date_index.append(i)

            df['is_next_day_holiday'] = 0
            for idx, value in enumerate(date_index):
                try:
                    if df[date_index[idx + 1]:date_index[idx + 3]]['is_holiday'].any() == 1:
                        df['is_next_day_holiday'][date_index[idx]:date_index[idx + 1]] = 1
                    else:
                        df['is_next_day_holiday'][date_index[idx]:date_index[idx + 1]] = 0
                except:
                    df['is_next_day_holiday'] = df['is_next_day_holiday'].fillna(0)

            df1 = df.filter(['year', 'month', 'day', 'dayofweek', 'is_holiday', 'is_next_day_holiday', 'PaxWeight'])

            # Define the columns you want to use as features
            features = ['year', 'month', 'day', 'dayofweek', 'is_holiday', 'is_next_day_holiday']

            # Create a PolynomialFeatures object with the desired degree
            poly = PolynomialFeatures(degree=5)

            # Fit and transform your data
            poly_features = poly.fit_transform(df1[features])

            # Create a list of column names for the polynomial features DataFrame
            poly_columns = []
            for i in range(poly_features.shape[1]):
                poly_columns.append(f'poly_{i}')

            # Create a new DataFrame with the polynomial features
            poly_df = pd.DataFrame(poly_features, columns=poly_columns)

            # Concatenate the original DataFrame with the polynomial features DataFrame
            data_temp = pd.concat([df1[features], poly_df], axis=1)
            df1 = pd.concat([data_temp, df1['PaxWeight']], axis=1)

            # Create a KBinsDiscretizer object with the desired number of bins and strategy
            # discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
            #
            # # Fit and transform your data
            # col_to_bin = list(df1.columns[:-1])
            # binned_data = pd.DataFrame(discretizer.fit_transform(df1[col_to_bin]), columns=col_to_bin)
            # df2 = pd.concat([binned_data, df1['PaxWeight']] , axis=1)

            ct1 = ColumnTransformer([
                ('num', StandardScaler(), list(df1.columns[:-1])),
                ('out', StandardScaler(), ['PaxWeight']),
            ])
            df_trans = ct1.fit_transform(X=df1)
            # Base estimators
            # base_estimators = [
            #     # ('ridge', Ridge()),
            #     # ('lasso', Lasso()),
            #     # ('lr', LinearRegression()),
            #     # ('thilsen', TheilSenRegressor(max_iter=20000)),
            #     # ('huber', HuberRegressor(max_iter=20000)),
            #     ('random_forest', RandomForestRegressor()),
            #     ('gbm', GradientBoostingRegressor()),
            #     ('ada', AdaBoostRegressor())
            # ]
            #
            # # Stacking Regressor
            # voting_regressor = VotingRegressor(
            #     estimators=base_estimators,
            #     # final_estimator=GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=6)
            # )

            # Pipeline
            # pipeline = Pipeline([
            #     # ('preprocessor', ct),
            #     # ('feature_selection', SelectFromModel(voting_regressor, prefit=False, importance_getter=voting_regressor.feature_importances_)),
            #     # ('voting_regressor', voting_regressor)
            #     ('gbm', GradientBoostingRegressor())
            # ])

            # KFold for cross-validation
            kfold = KFold(n_splits=5)  # shuffle=True

            # Hyperparameter grid
            param_grid = {
                # 'stacked_regressor__ridge__alpha': np.logspace(-3, 0, 2),
                # 'stacked_regressor__lasso__alpha': np.logspace(-3, 0, 2),
                # 'stacked_regressor__random_forest__n_estimators': [100, 200, 500],
                # 'stacked_regressor__random_forest__max_depth': [2,3,4,5],
                'n_estimators': [500, 1000],
                'learning_rate': [0.01],
                'max_depth': [2, 3, 4, 5, 6, 7],
                # 'stacked_regressor__ada__n_estimators': [10, 50, 100],
                # 'stacked_regressor__ada__learning_rate': [0.01]
            }

            # GridSearchCV
            grid_search = GridSearchCV(
                GradientBoostingRegressor(), param_grid, scoring='neg_mean_squared_error', cv=kfold, n_jobs=3, verbose=1
            )

            # plt.figure(figsize=(10, 6))
            # sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
            # plt.show()
            # plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')

            test_len = 1
            if len(df_trans) >= 2:
                x_train = df_trans[:-test_len, :-1]
                y_train = df_trans[:-test_len, -1:]
                x_test = df_trans[-test_len:, :-1]
                # y_test = df_trans.iloc[-test_len:, -1:]
            else:
                requests.post(url='http://192.168.115.10:8081/api/Robots/CreateFlightLoadEstimate',
                              json={
                                  "fkFlightInformation": PkFlightInformation,
                                  "load": -1,
                                  "FlightCount": None
                              },
                              headers={'Authorization': f'Bearer {token}',
                                       'Content-type': 'application/json',
                                       })
                continue
            if len(df) >= 20:
                grid_search.fit(x_train, y_train)
                model = grid_search.best_estimator_
            else:
                model = GradientBoostingRegressor(
                    random_state=10,
                    learning_rate=.01,
                    max_depth=2,
                    n_estimators=10000,
                    n_iter_no_change=800,
                    # verbose=1,
                    # max_leaf_nodes=3,
                )

            selector = SelectFromModel(model, prefit=False).fit(x_train,y_train)
            x_train_selected = selector.transform(x_train)
            x_test_selected = selector.transform(x_test)
            model = model.fit(x_train_selected, y_train)
            y_pred = model.predict(x_test_selected)
            y_pred = ct1.named_transformers_['out'].inverse_transform(y_pred.reshape(-1, 1))

            token = api_token_handler()
            flight_info = requests.post(url='http://192.168.115.10:8081/api/FlightInformation/GetFlightInformation',
                                        json={"pkFlightInformation": PkFlightInformation},
                                        headers={'Authorization': f'Bearer {token}',
                                                 'Content-type': 'application/json'})

            result_data = {
                "fkFlightInformation": PkFlightInformation,
                "load": y_pred[0][0],
                "FlightCount": df_len,
                "FlightDate": ' '.join(json.loads(flight_info.text)['date'].split('T'))
            }
            api_result = requests.post(url='http://192.168.115.10:8081/api/Robots/CreateFlightLoadEstimate',
                                       json=result_data,
                                       headers={'Authorization': f'Bearer {token}',
                                                'Content-type': 'application/json',
                                                })
            if not json.loads(api_result.text)['success']:
                print(f'y_pred:{y_pred} with {PkFlightInformation} did not recorded in DB.')
        except:
            token = api_token_handler()
            result_data = {
                "fkFlightInformation": PkFlightInformation,
                "load": -1,
                "FlightCount": 0
            }
            requests.post(url='http://192.168.115.10:8081/api/Robots/CreateFlightLoadEstimate',
                          json=result_data,
                          headers={'Authorization': f'Bearer {token}',
                                   'Content-type': 'application/json',
                                   })

    # for input_data_dict in input_data_list:
