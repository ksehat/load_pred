import pandas as pd
import requests
import json
import copy
import pickle
import numpy as np
from tensorflow import keras
from functions import api_token_handler


def load_pred_pretrained_model(df_dict):
    df = pd.DataFrame(df_dict)
    df.rename(columns={'Is_holiday': 'is_holiday', 'Route': 'FlightRoute', 'GregorianDate': 'Departure'}, inplace=True)
    df['Departure'] = pd.to_datetime(df['Departure'])
    df['FlightRoute'] = df['FlightRoute'].str.replace('>', '-')
    df['PaxWeight'] = 0

    df_old = pd.read_excel('data/df.xlsx')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    for i in range(len(df)):
        try:
            pkFlightInformation = df.iloc[i:i + 1, :]['PkFlightInformation'].values[0]
            flightdate = df.iloc[i:i + 1, :]['Departure'].values[0]
            df0 = pd.DataFrame()
            df0 = pd.concat([df_old, df.iloc[i:i + 1, :]], axis=0, ignore_index=True)
            df0.drop(['GregorianDate', 'PkFlightInformation'], axis=1, inplace=True)

            df0['FlightRoute'] = le.transform(df0['FlightRoute'])

            df0['year'] = np.array(pd.DatetimeIndex(df0['Departure']).year)
            df0['month'] = np.array(pd.DatetimeIndex(df0['Departure']).month)
            df0['day'] = np.array(pd.DatetimeIndex(df0['Departure']).day)
            df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['Departure']).dayofweek)
            df0['hour'] = np.array(pd.DatetimeIndex(df0['Departure']).hour)

            shift_num = 10
            df_temp0 = copy.deepcopy(df0)
            for kan1 in range(shift_num):
                df0 = pd.concat(
                    [df0, df_temp0.groupby('FlightRoute').shift(periods=kan1 + 1).add_suffix(f'_shifted{kan1 + 1}')],
                    axis=1)

            df0.dropna(inplace=True)

            filtered_columns_list = ['year', 'month', 'day', 'dayofweek', 'hour', 'FlightRoute', 'is_holiday']
            all_org_rows_list = filtered_columns_list + ['PaxWeight']
            for kan2 in range(shift_num):
                filtered_columns_list_temp = [x + f'_shifted{kan2 + 1}' for x in all_org_rows_list]
                for x in filtered_columns_list_temp:
                    filtered_columns_list.append(x)
            filtered_columns_list.append('PaxWeight')

            df1 = df0.filter(filtered_columns_list)

            model = keras.models.load_model('my_model.h5')
            x_result = df1.iloc[-1, :-1].values.reshape(1, -1)
            y_pred = model.predict(x_result.reshape((x_result.shape[0], x_result.shape[1], 1)))
            token = api_token_handler()
            result_data = {
                "fkFlightInformation": int(pkFlightInformation),
                "load": float(y_pred[0][0]),
                "flightCount": 1,
                "flightDate": str(flightdate).split('.')[0]
            }
            api_result = requests.post(url='http://192.168.115.10:8081/api/Robots/CreateFlightLoadEstimate',
                           json=result_data,
                           headers={'Authorization': f'Bearer {token}',
                                    'Content-type': 'application/json',
                                    })
            if not json.loads(api_result.text)['success']:
                print(f'y_pred:{y_pred} with {pkFlightInformation} did not recorded in DB.')
        except:
            result_data = {
                "fkFlightInformation": int(pkFlightInformation),
                "load": 0,
                "flightCount": 1,
                "flightDate": str(flightdate).split('.')[0]
            }
            requests.post(url='http://192.168.115.10:8081/api/Robots/CreateFlightLoadEstimate',
                          json=result_data,
                          headers={'Authorization': f'Bearer {token}',
                                   'Content-type': 'application/json',
                                   })
