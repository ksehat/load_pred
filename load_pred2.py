import copy
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.ensemble import StackingRegressor
from functions import api_token_handler


def mean_load_pred2(input_data_list):
    for input_data_dict in input_data_list:
        try:
            df = pd.DataFrame(input_data_dict['GetAllFlightsWithSameConditionRobotResponseItemViewModels'])
            PkFlightInformation = input_data_dict['PkFlightInformation']
            df['year'] = np.array(pd.DatetimeIndex(df['GregorianDate']).year)
            df['month'] = np.array(pd.DatetimeIndex(df['GregorianDate']).month)
            df['day'] = np.array(pd.DatetimeIndex(df['GregorianDate']).day)
            df = df[df['PaxWeight'] != 0]

            last_date = df['GregorianDate'][0]
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

            df1 = df.filter(['year', 'month', 'day', 'is_holiday', 'is_next_day_holiday', 'PaxWeight'])

            ct1 = ColumnTransformer([
                ('num', StandardScaler(), ['year', 'month', 'day']),
                ('cat', OneHotEncoder(), ['is_holiday', 'is_next_day_holiday']),
                ('out', StandardScaler(), ['PaxWeight']),
            ])
            df_trans = ct1.fit_transform(X=df1)
            # Base estimators
            base_estimators = [
                ('ridge', Ridge()),
                ('lasso', Lasso()),
                ('lr', LinearRegression()),
                ('random_forest', RandomForestRegressor()),
                ('gbm', GradientBoostingRegressor()),
                ('ada', AdaBoostRegressor())
            ]

            # Stacking Regressor
            stacked_regressor = StackingRegressor(
                estimators=base_estimators,
                final_estimator=LinearRegression()
            )

            # Pipeline
            pipeline = Pipeline([
                # ('preprocessor', ct),
                ('stacked_regressor', stacked_regressor)
            ])

            # KFold for cross-validation
            kfold = KFold(n_splits=5, random_state=None) #shuffle=True

            # Hyperparameter grid
            param_grid = {
                'stacked_regressor__ridge__alpha': np.logspace(-3, 0, 2),
                'stacked_regressor__lasso__alpha': np.logspace(-3, 0, 2),
                # 'stacked_regressor__random_forest__n_estimators': [100, 200, 500],
                # 'stacked_regressor__random_forest__max_depth': [2,3,4,5],
                'stacked_regressor__gbm__n_estimators': [100,200, 500, 1000],
                # 'stacked_regressor__gbm__learning_rate': [0.01],
                'stacked_regressor__gbm__max_depth': [2,3,4,5],
                # 'stacked_regressor__ada__n_estimators': [10, 50, 100],
                # 'stacked_regressor__ada__learning_rate': [0.01]
            }

            # GridSearchCV
            grid_search = GridSearchCV(
                pipeline, param_grid, scoring='neg_mean_squared_error', cv=kfold, n_jobs=-1, verbose=1
            )

            # plt.figure(figsize=(10, 6))
            # sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
            # plt.show()
            # plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')

            test_len = 1
            x_train = df_trans[:-test_len, :-1]
            y_train = df_trans[:-test_len, -1:]
            x_test = df_trans[-test_len:, :-1]
            y_test = df_trans[-test_len:, -1:]
            if len(df) == 0:
                requests.post(url='http://192.168.115.10:8083/api/FlightInformation/CreateFlightLoadEstimate',
                              json={
                                  "fkFlightInformation": PkFlightInformation,
                                  "load": -1
                              },
                              headers={'Authorization': f'Bearer {token}',
                                       'Content-type': 'application/json',
                                       })
                continue
            if len(df) >= 20:
                grid_search.fit(x_train, y_train)
                model = grid_search.best_estimator_
                # model1 = GradientBoostingRegressor(
                #     random_state=10,
                #     learning_rate=.01,
                #     max_depth=2,
                #     n_estimators=10000,
                #     n_iter_no_change=800,
                #     # verbose=1,
                #     # max_leaf_nodes=3,
                # )
                # # model2 = HuberRegressor()
                # model3 = RandomForestRegressor(random_state=10)
                # model4 = AdaBoostRegressor(random_state=10, n_estimators=100, learning_rate=0.01)
                # model5 = LinearRegression()
                #
                # estimators = [('gbr', model1), ('forest', model3), ('ada', model4), ('linear', model5)]
                # model = StackingRegressor(estimators=estimators, final_estimator=AdaBoostRegressor(random_state=3))
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
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            # print(np.mean(np.abs(y_pred - y_test)), md)
            y_pred = ct1.named_transformers_['out'].inverse_transform(y_pred.reshape(-1,1))
            token = api_token_handler()
            result_data = {
                "fkFlightInformation": PkFlightInformation,
                "load": y_pred[0][0]
            }
            api_result = requests.post(url='http://192.168.115.10:8081/api/FlightInformation/CreateFlightLoadEstimate',
                                       json=result_data,
                                       headers={'Authorization': f'Bearer {token}',
                                                'Content-type': 'application/json',
                                                })
            if not json.loads(api_result.text)['success']:
                print(f'y_pred:{y_pred} did not recorded in DB.')
        except:
            requests.post(url='http://192.168.115.10:8081/api/FlightInformation/CreateFlightLoadEstimate',
                          json=result_data,
                          headers={'Authorization': f'Bearer {token}',
                                   'Content-type': 'application/json',
                                   })

