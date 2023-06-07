import pandas as pd
import requests
import json
import copy
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from functions import api_token_handler


def baggage_pred_pretrained_model():
    token = api_token_handler()
    df_future = pd.DataFrame(
        json.loads(requests.get(url='http://192.168.115.10:8083/api/FlightBaggageEstimate/GetAllFutureFlightsBaggage',
                                headers={'Authorization': f'Bearer {token}',
                                         'Content-type': 'application/json',
                                         }
                                ).text)['getAllFutureFlightsBaggageResponseItemViewModels'])
    df_past = pd.DataFrame(
        json.loads(requests.get(url='http://192.168.115.10:8083/api/FlightBaggageEstimate/GetAllPastFlightsBaggage',
                                headers={'Authorization': f'Bearer {token}',
                                         'Content-type': 'application/json',
                                         }
                                ).text)['getAllPastFlightsBaggageResponseItemViewModels'])
    df_future.rename(columns={'Is_holiday': 'is_holiday', 'Route': 'route', 'Departure': 'departure',
                              'PkFlightInformation': 'pkFlightInformation'}, inplace=True)
    df_future['departure'] = pd.to_datetime(df_future['departure'])
    df_future['paxWeight'] = 0

    token = api_token_handler()

    df_past = pd.DataFrame(
        json.loads(requests.get(url='http://192.168.115.10:8083/api/FlightLoadEstimate/GetAllPastFlightsLoad',
                                headers={'Authorization': f'Bearer {token}',
                                         'Content-type': 'application/json',
                                         }
                                ).text)['getAllPastFlightsLoadResponseItemViewModels'])

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    df_future['route'] = df_future['route'].apply(lambda x: x.replace(">", "-"))
    df_past['route'] = df_past['route'].apply(lambda x: x.replace(">", "-"))

    for i in range(len(df_future)):
        try:
            pkFlightInformation = df_future.iloc[i:i + 1, :]['pkFlightInformation'].values[0]
            flightdate = df_future.iloc[i:i + 1, :]['departure'].values[0]

            df0 = pd.concat([df_past, df_future.iloc[i:i + 1, :]], axis=0, ignore_index=True)
            df0.drop(['pkFlightInformation'], axis=1, inplace=True)

            df0['route'] = df0['route'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

            df0['year'] = np.array(pd.DatetimeIndex(df0['departure']).year)
            df0['month'] = np.array(pd.DatetimeIndex(df0['departure']).month)
            df0['day'] = np.array(pd.DatetimeIndex(df0['departure']).day)
            df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['departure']).dayofweek)
            df0['hour'] = np.array(pd.DatetimeIndex(df0['departure']).hour)

            shift_num = 20
            df_temp0 = copy.deepcopy(df0)
            for kan1 in range(shift_num):
                df0 = pd.concat(
                    [df0, df_temp0.groupby('route').shift(periods=kan1 + 1).add_suffix(f'_shifted{kan1 + 1}')],
                    axis=1)

            df0.dropna(inplace=True)

            filtered_columns_list = ['year', 'month', 'day', 'dayofweek', 'hour', 'route', 'is_holiday']
            all_org_rows_list = filtered_columns_list + ['paxWeight']
            for kan2 in range(shift_num):
                filtered_columns_list_temp = [x + f'_shifted{kan2 + 1}' for x in all_org_rows_list]
                for x in filtered_columns_list_temp:
                    filtered_columns_list.append(x)
            filtered_columns_list.append('paxWeight')

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
            api_result = requests.post(url='http://192.168.115.10:8083/api/FlightLoadEstimate/CreateFlightLoadEstimate',
                                       json=result_data,
                                       headers={'Authorization': f'Bearer {token}',
                                                'Content-type': 'application/json',
                                                })
            if not json.loads(api_result.text)['success']:
                print(f'y_pred:{y_pred} with {pkFlightInformation} did not recorded in DB.')
        except:
            token = api_token_handler()
            result_data = {
                "fkFlightInformation": int(pkFlightInformation),
                "load": 0,
                "flightCount": 1,
                "flightDate": str(flightdate).split('.')[0]
            }
            requests.post(url='http://192.168.115.10:8083/api/FlightLoadEstimate/CreateFlightLoadEstimate',
                          json=result_data,
                          headers={'Authorization': f'Bearer {token}',
                                   'Content-type': 'application/json',
                                   })


baggage_pred_pretrained_model()