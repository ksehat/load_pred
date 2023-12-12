import pandas as pd
import requests
import json
from functions import api_token_handler


def baggage_pred_pretrained_model():
    token = api_token_handler()
    df_past = pd.DataFrame(
        json.loads(requests.get(url='http://192.168.115.10:8081/api/FlightBaggageEstimate/GetAllPastFlightsBaggage',
                                headers={'Authorization': f'Bearer {token}',
                                         'Content-type': 'application/json',
                                         }
                                ).text)['getAllPastFlightsBaggageResponseItemViewModels']).sort_values(by='departure')
    return df_past

df_past = baggage_pred_pretrained_model()
print(df_past)