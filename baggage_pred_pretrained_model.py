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
import joblib



def baggage_pred_pretrained_model():
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

    with open('baggage_deployed_models\label_encoder_baggage.pkl', 'rb') as f:
        le = pickle.load(f)

    df_future['route'] = df_future['route'].apply(lambda x: x.replace("-", ">"))
    df_past['route'] = df_past['route'].apply(lambda x: x.replace("-", ">"))

    for i in range(len(df_future)):
        try:
            pkFlightInformation = df_future.iloc[i:i + 1, :]['pkFlightInformation'].values[0]
            flightdate = df_future.iloc[i:i + 1, :]['departure'].values[0]

            df0 = pd.concat([df_past, df_future.iloc[i:i + 1, :]], axis=0, ignore_index=True)
            df0.drop(['pkFlightInformation', 'paxWeight', 'payLoad'], axis=1, inplace=True)

            df0['route'] = df0['route'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else len(le.classes_)+1)

            df0['baggage'] = df0['baggage'].str.split('/', expand=True)[1]
            df0['baggage'] = df0['baggage'].str.split(' ', expand=True)[0]
            df0['baggage'] = df0['baggage'].astype(float)

            df0['year'] = np.array(pd.DatetimeIndex(df0['departure']).year)
            df0['month'] = np.array(pd.DatetimeIndex(df0['departure']).month)
            df0['day'] = np.array(pd.DatetimeIndex(df0['departure']).day)
            df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['departure']).dayofweek)
            df0['hour'] = np.array(pd.DatetimeIndex(df0['departure']).hour)
            df0['is_holiday'][(np.array(pd.DatetimeIndex(df0['departure']).day_name()) == 'Friday')] = 1
            df0['is_holiday'][(np.array(pd.DatetimeIndex(df0['departure']).day_name()) == 'Thursday')] = 1

            df0['departure'] = pd.to_datetime(df0['departure'])
            df0.sort_values(by='departure', inplace=True)
            df0.reset_index(drop=True, inplace=True)

            holidays = df0.loc[df0['is_holiday'] == 1, 'departure']
            df0['days_until_holiday'] = holidays.reindex(df0.index, method='bfill').dt.date - df0['departure'].dt.date
            df0['days_until_holiday'] = pd.to_timedelta(df0['days_until_holiday']).dt.days

            df0['route'] = le.fit_transform(df0['route'])
            df0.drop(['departure'], inplace=True, axis=1)

            shift_num = 10
            df_temp0 = copy.deepcopy(df0)
            for kan1 in range(shift_num):
                df0 = pd.concat(
                    [df0, df_temp0.groupby('route').shift(periods=kan1 + 1).add_suffix(f'_shifted{kan1 + 1}')],
                    axis=1)

            df0.dropna(inplace=True)

            col = df0.pop('baggage')
            df0.insert(len(df0.columns), 'baggage', col)

            model1 = keras.models.load_model('baggage_deployed_models/baggage_model1.h5')
            model2 = joblib.load('baggage_deployed_models/baggage_model2.sav')
            model3 = keras.models.load_model('baggage_deployed_models/baggage_model3.h5')

            df1 = copy.deepcopy(np.array(df0))
            x_result = df1[-1, :-1]
            y_pred1 = model1.predict(x_result.reshape(1,-1))
            y_pred2 = model2.predict(x_result.reshape(1,-1))
            x_result_final = np.concatenate((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)), axis=1)
            y_pred_final = model3.predict(x_result_final.reshape(1,-1))

            token = api_token_handler()
            result_data = {
                "fkFlightInformation": int(pkFlightInformation),
                "load": float(y_pred_final[0][0]),
                "flightCount": 1,
                "flightDate": str(flightdate).split('.')[0].replace('T',' ')
            }
            api_result = requests.post(url='http://192.168.115.10:8081/api/FlightBaggageEstimate/CreateFlightBaggageEstimate',
                                       json=result_data,
                                       headers={'Authorization': f'Bearer {token}',
                                                'Content-type': 'application/json',
                                                })
            if not json.loads(api_result.text)['success']:
                print(f'y_pred:{y_pred_final} with {pkFlightInformation} did not recorded in DB.')
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
