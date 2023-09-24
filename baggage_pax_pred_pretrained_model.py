import time
import pandas as pd
import requests
import json
import copy
import pickle
import numpy as np
import keras
from functions import api_token_handler
import joblib
import schedule
from joblib import load
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler


def apply_label_dict(column):
    label_dict = load('artifacts/baggage_pax/baggage_pax_deployed_models/label_dict.joblib')
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


def get_last_5_baggage(df, reverse_route, date):
    mask = (df['route'] == reverse_route) & (df['departure'] < date)
    return df.loc[mask, 'baggage'].tail(5).tolist()

def get_last_5_pax(df, reverse_route, date):
    mask = (df['route'] == reverse_route) & (df['departure'] < date)
    return df.loc[mask, 'paxWeight'].tail(5).tolist()


def push_to_db_baggage(y_pred_final, pkFlightInformation, flightdate, sales_weight, model):
    try:
        y_pred_final = y_pred_final[0][0]
    except:
        y_pred_final = y_pred_final[0]
    print(y_pred_final)
    print(
        f"For flight with fkInfo: {int(pkFlightInformation)} at {str(flightdate).split('.')[0].replace('T', ' ')}.")
    token = api_token_handler()
    result_data = {
        "fkFlightInformation": int(pkFlightInformation),
        "load": float(y_pred_final),
        "flightCount": 1,
        "flightDate": str(flightdate).split('.')[0].replace('T', ' '),
        "salesWeight": float(sales_weight),
        "Model": model
    }
    api_result = requests.post(
        url='http://192.168.115.10:8081/api/FlightBaggageEstimate/CreateFlightBaggageEstimate',
        json=result_data,
        headers={'Authorization': f'Bearer {token}',
                 'Content-type': 'application/json',
                 })


def push_to_db_pax(y_pred_final, pkFlightInformation, flightdate, sales_weight, model):
    try:
        y_pred_final = y_pred_final[0][0]
    except:
        y_pred_final = y_pred_final[0]
    print(y_pred_final)
    print(
        f"For flight with fkInfo: {int(pkFlightInformation)} at {str(flightdate).split('.')[0].replace('T', ' ')}.")
    token = api_token_handler()
    result_data = {
        "fkFlightInformation": int(pkFlightInformation),
        "load": float(y_pred_final),
        "flightCount": 1,
        "flightDate": str(flightdate).split('.')[0].replace('T', ' '),
        "salesWeight": float(sales_weight),
        "Model": model
    }
    api_result = requests.post(
        url='http://192.168.115.10:8081/api/FlightLoadEstimate/CreateFlightLoadEstimate',
        json=result_data,
        headers={'Authorization': f'Bearer {token}',
                 'Content-type': 'application/json',
                 })


def baggage_pax_pred_pretrained_model():
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

    df_future['route'] = df_future['route'].apply(lambda x: x.replace("-", ">"))
    df_past['route'] = df_past['route'].apply(lambda x: x.replace("-", ">"))

    model1 = keras.models.load_model('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model1.h5')
    model1.load_weights(
        'C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage_pax/baggage_pax_similarity_training_weights\model1/weights_epoch128.h5')
    model2_baggage = joblib.load('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model2.sav')
    model2_pax = joblib.load('artifacts/baggage_pax/baggage_pax_deployed_models/pax_model2.sav')
    model3 = keras.models.load_model('artifacts/baggage_pax/baggage_pax_deployed_models/baggage_model3.h5')
    model3.load_weights(
        'C:/Users\Administrator\Desktop\Projects\member_pred/artifacts/baggage_pax/baggage_pax_similarity_training_weights\model3/weights_epoch47.h5')
    model_for_new_routes_baggage = joblib.load('artifacts/baggage_pax/baggage_pax_deployed_models/hgbr_baggage.pkl')
    model_for_new_routes_pax = joblib.load('artifacts/baggage_pax/baggage_pax_deployed_models/hgbr_paxWeight.pkl')

    for i in range(len(df_future)):
        try:
            pkFlightInformation = df_future.iloc[i:i + 1, :]['pkFlightInformation'].values[0]
            flightdate = df_future.iloc[i:i + 1, :]['departure'].values[0]
            sales_weight = df_future.iloc[i:i + 1, :]['salesWeight'].values[0]

            df0 = pd.concat([df_past, df_future.iloc[i:i + 1, :]], axis=0, ignore_index=True)
            df0.drop(['pkFlightInformation', 'salesWeight'], axis=1, inplace=True)

            df0['route'] = apply_label_dict(df0['route'])

            df0['baggage'] = df0['baggage'].str.split('/', expand=True)[1]
            df0['baggage'] = df0['baggage'].str.split(' ', expand=True)[0]
            df0['baggage'] = df0['baggage'].astype(float)

            df0['year'] = np.array(pd.DatetimeIndex(df0['departure']).year)
            df0['month'] = np.array(pd.DatetimeIndex(df0['departure']).month)
            df0['day'] = np.array(pd.DatetimeIndex(df0['departure']).day)
            df0['dayofweek'] = np.array(pd.DatetimeIndex(df0['departure']).dayofweek)
            df0['hour'] = np.array(pd.DatetimeIndex(df0['departure']).hour)
            df0['quarter'] = np.array(pd.DatetimeIndex(df0['departure']).quarter)

            # create a function to get the last 5 values of the baggage column for the reverse route
            last_5_values_baggage = df0.apply(lambda x: get_last_5_baggage(df0, -x['route'], x['departure']), axis=1)
            for kan4 in range(5):
                df0[f'reverse_baggage_{kan4 + 1}'] = last_5_values_baggage.apply(lambda x: x[kan4] if len(x) > kan4 else None)

            last_5_values_pax = df0.apply(lambda x: get_last_5_pax(df0, -x['route'], x['departure']), axis=1)
            for kan4 in range(5):
                df0[f'reverse_pax_{kan4 + 1}'] = last_5_values_pax.apply(lambda x: x[kan4] if len(x) > kan4 else None)

            df0.drop(['departure'], inplace=True, axis=1)

            col = df0.pop('baggage')
            df0.insert(len(df0.columns), 'baggage', col)
            col = df0.pop('paxWeight')
            df0.insert(len(df0.columns), 'paxWeight', col)

            shift_num = 15
            df_temp0 = copy.deepcopy(df0)
            for kan1 in range(shift_num):
                df0 = pd.concat(
                    [df0, df_temp0.groupby('route').shift(periods=kan1 + 1).add_suffix(f'_shifted{kan1 + 1}')],
                    axis=1)

            col = df0.pop('baggage')
            df0.insert(len(df0.columns), 'baggage', col)
            col = df0.pop('paxWeight')
            df0.insert(len(df0.columns), 'paxWeight', col)

            if any(df0.iloc[-1:, :-2]) == None:
                y_pred_final_baggage = model_for_new_routes_baggage.predict(df0.iloc[-1, :-2])
                y_pred_final_pax = model_for_new_routes_pax.predict(df0.iloc[-1, :-2])
                try:
                    push_to_db_baggage(y_pred_final_baggage, pkFlightInformation, flightdate, sales_weight, 'hgbr_baggage_pax')
                    push_to_db_pax(y_pred_final_pax, pkFlightInformation, flightdate, sales_weight, 'hgbr_baggage_pax')
                    continue
                except:
                    print(f'Error: y_pred:{y_pred_final} with {pkFlightInformation} did not recorded in DB.')
                    continue

            df0_temp = df0.iloc[-1:, :]
            df0.dropna(inplace=True)
            df0 = pd.concat([df0, df0_temp], axis=0)
            df1 = copy.deepcopy(df0)
            df1.reset_index(drop=True, inplace=True)

            n = 3  # number of similar rows to find
            similarity_columns = list(df1.columns)[:-2]
            df1_np = df1[similarity_columns].to_numpy()

            # Find the n most similar rows for each row
            most_similar_rows = []
            baggage_indice_in_df1 = len(df1.columns)-2
            pax_indice_in_df1 = len(df1.columns)-1
            for i2 in range(n + 1, len(df1)):
                # Build a KDTree with the rows before the current row
                if i2 >= 100:
                    tree = KDTree(df1_np[i2 - 100:i2 + 1])
                else:
                    tree = KDTree(df1_np[:i2 + 1])

                # Find the n most similar rows
                dist, ind = tree.query(df1_np[i2:i2 + 1], k=n + 1)
                most_similar_indices = ind[0][:n + 1] + i2 - 100 if i2 >= 100 else ind[0][:n + 1]
                most_similar_rows.append(df1.iloc[most_similar_indices].stack().to_frame().reset_index(drop=True).T)

            # Create a new dataframe with the most similar rows as new columns
            arr1 = np.stack([df.to_numpy() for df in most_similar_rows[:-1]], axis=0).squeeze()
            arr2 = np.delete(arr1, baggage_indice_in_df1, axis=1)
            arr2 = np.delete(arr2, pax_indice_in_df1, axis=1)
            arr3 = np.concatenate((arr2, np.array(most_similar_rows[-1])))

            ss = load(open('artifacts/baggage_pax/baggage_pax_deployed_models/scaler.pkl', 'rb'))
            arr4 = ss.transform(arr3)

            x_result = arr4[-1].reshape(1, -1)
            y_pred1 = model1.predict(x_result)
            y_pred2_baggage = model2_baggage.predict(x_result)
            y_pred2_pax = model2_pax.predict(x_result)
            x_result_final_baggage = np.concatenate((y_pred1[:,0].reshape(-1, 1), y_pred2_baggage.reshape(-1, 1)), axis=1)
            x_result_final_pax = np.concatenate((y_pred1[:,1].reshape(-1, 1), y_pred2_pax.reshape(-1, 1)), axis=1)
            x_result_final = np.concatenate((x_result_final_baggage, x_result_final_pax), axis=1)
            y_pred_final = model3.predict(x_result_final.reshape(1, -1))

            try:
                push_to_db_baggage(y_pred_final[:,0], pkFlightInformation, flightdate, sales_weight, 'deep_BP')
                push_to_db_pax(y_pred_final[:,1], pkFlightInformation, flightdate, sales_weight, 'deep_BP')
                continue
            except:
                print(f'Error: y_pred:{y_pred_final} with {pkFlightInformation} did not recorded in DB.')
                continue

        except:
            if df0.iloc[-1, :-1].any():
                y_pred_final_baggage = model_for_new_routes_baggage.predict(df0.iloc[-1, :-2])
                y_pred_final_pax = model_for_new_routes_pax.predict(df0.iloc[-1, :-2])
                try:
                    push_to_db_baggage(y_pred_final_baggage, pkFlightInformation, flightdate, sales_weight,
                                       'hgbr_baggage_pax')
                    push_to_db_pax(y_pred_final_pax, pkFlightInformation, flightdate, sales_weight, 'hgbr_baggage_pax')
                    continue
                except:
                    print(f'Error: y_pred:{y_pred_final} with {pkFlightInformation} did not recorded in DB.')
                    continue
            print(
                f"Non payload error for flight with fkInfo: {int(pkFlightInformation)} at {str(flightdate).split('.')[0].replace('T', ' ')}.")
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
