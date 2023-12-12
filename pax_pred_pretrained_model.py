import pandas as pd
import requests
import json
import copy
import numpy as np
import keras
from functions import api_token_handler
import joblib
from joblib import load


def apply_label_dict(column):
    label_dict = load('pax_deployed_models/label_dict.joblib')
    encoded_column = []
    for route in column:
        origin, destination = route.split('>')
        reverse_route = f'{destination}>{origin}'
        if route in label_dict:
            encoded_column.append(label_dict[route])
        elif reverse_route in label_dict:
            encoded_column.append(-label_dict[reverse_route])
        else:
            encoded_column.append(max(list(label_dict.values())))
    # or any other default value
    return encoded_column


def get_last_5(df, reverse_route, date):
    mask = (df['route'] == reverse_route) & (df['departure'] < date)
    return df.loc[mask, 'paxWeight'].tail(5).tolist()


def pax_pred_pretrained_model(who_is_calling=None):
    token = api_token_handler()
    df_past = pd.DataFrame(
        json.loads(requests.get(url='http://192.168.115.10:8081/api/FlightBaggageEstimate/GetAllPastFlightsBaggage',
                                headers={'Authorization': f'Bearer {token}',
                                         'Content-type': 'application/json',
                                         }
                                ).text)['getAllPastFlightsBaggageResponseItemViewModels']).sort_values(by='departure')
    df_future = pd.DataFrame(
        json.loads(requests.get(url='http://192.168.115.10:8081/api/FlightBaggageEstimate/GetAllFutureFlightsBaggage',
                                headers={'Authorization': f'Bearer {token}',
                                         'Content-type': 'application/json',
                                         }
                                ).text)['getAllFutureFlightsBaggageResponseItemViewModels']).sort_values(by='departure')

    df_future['departure'] = pd.to_datetime(df_future['departure'])
    df_past['departure'] = pd.to_datetime(df_past['departure'])

    df_past.sort_values(by='departure', inplace=True)
    df_past.reset_index(drop=True, inplace=True)
    df_future.sort_values(by='departure', inplace=True)
    df_future.reset_index(drop=True, inplace=True)

    df_future['route'] = df_future['route'].apply(lambda x: x.replace("-", ">"))
    df_past['route'] = df_past['route'].apply(lambda x: x.replace("-", ">"))

    if who_is_calling != 'pax_train':
        df_future.drop('baggage', axis=1, inplace=True)
        df_past.drop('baggage', axis=1, inplace=True)

    for i in range(len(df_future)):
        try:
            pkFlightInformation = df_future.iloc[i:i + 1, :]['pkFlightInformation'].values[0]
            flightdate = df_future.iloc[i:i + 1, :]['departure'].values[0]

            df0 = pd.concat([df_past, df_future.iloc[i:i + 1, :]], axis=0, ignore_index=True)
            df0.drop(['pkFlightInformation', 'salesWeight'], axis=1, inplace=True)

            df0['route'] = apply_label_dict(df0['route'])

            df0['year'] = np.array(pd.DatetimeIndex(df0['departure']).year)
            df0['month'] = np.array(pd.DatetimeIndex(df0['departure']).month)
            df0['day'] = np.array(pd.DatetimeIndex(df0['departure']).day)
            df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['departure']).dayofweek)
            df0['hour'] = np.array(pd.DatetimeIndex(df0['departure']).hour)
            df0['quarter'] = np.array(pd.DatetimeIndex(df0['departure']).quarter)

            # create a function to get the last 5 values of the baggage column for the reverse route
            last_5_values = df0.apply(lambda x: get_last_5(df0, -x['route'], x['departure']), axis=1)
            for kan4 in range(5):
                df0[f'reverse_paxWeight_{kan4 + 1}'] = last_5_values.apply(lambda x: x[kan4] if len(x) > kan4 else None)

            df0.drop(['departure'], inplace=True, axis=1)

            col = df0.pop('paxWeight')
            df0.insert(len(df0.columns), 'paxWeight', col)

            shift_num = 15
            df_temp0 = copy.deepcopy(df0)
            for kan1 in range(shift_num):
                df0 = pd.concat(
                    [df0, df_temp0.groupby('route').shift(periods=kan1 + 1).add_suffix(f'_shifted{kan1 + 1}')],
                    axis=1)

            df0.drop('paxWeight', inplace=True, axis=1)
            df0.dropna(inplace=True)
            df1 = copy.deepcopy(np.array(df0))

            model1 = keras.models.load_model('pax_deployed_models/pax_model1.h5')
            model1.load_weights(
                'E:\Projects\member_pred/pax_training_weights\model1/weights_epoch513.h5')
            model2 = joblib.load('pax_deployed_models/pax_model2.sav')
            model3 = keras.models.load_model('pax_deployed_models/pax_model3.h5')
            model3.load_weights(
                'E:\Projects\member_pred/pax_training_weights\model3/weights_epoch733.h5')

            if who_is_calling == 'pax_train':
                x_result_past = df1[:-1]
                y_pred1_past = model1.predict(x_result_past)
                y_pred2_past = model2.predict(x_result_past)
                x_result_final_past = np.concatenate((y_pred1_past.reshape(-1, 1), y_pred2_past.reshape(-1, 1)), axis=1)
                df_past2 = copy.deepcopy(df_past)
                df_past2['pax_train'] = model3.predict(x_result_final_past)
                df0['baggage'] = df_past['baggage']
                return df0['pax_train']

            x_result = df1[-1].reshape(1, -1)
            y_pred1 = model1.predict(x_result)
            y_pred2 = model2.predict(x_result)
            x_result_final = np.concatenate((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)), axis=1)
            y_pred_final = model3.predict(x_result_final.reshape(1, -1))

            if y_pred_final[0][0] >= 40:
                print(y_pred_final[0][0])
                token = api_token_handler()
                result_data = {
                    "fkFlightInformation": int(pkFlightInformation),
                    "load": float(y_pred_final[0][0]),
                    "flightCount": 1,
                    "flightDate": str(flightdate).split('.')[0].replace('T', ' '),
                }
                api_result = requests.post(
                    url='http://192.168.115.10:8081/api/FlightLoadEstimate/CreateFlightLoadEstimate',
                    json=result_data,
                    headers={'Authorization': f'Bearer {token}',
                             'Content-type': 'application/json',
                             })
                if not json.loads(api_result.text)['success']:
                    print(f'y_pred:{y_pred_final} with {pkFlightInformation} did not recorded in DB.')
            if who_is_calling == 'pax_train':
                return y_pred_final[0][0]
        except:
            print('An error occured.')
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
    print('Prediction job is done.')
