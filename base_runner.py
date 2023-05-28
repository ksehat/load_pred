import json
import requests
from functions import api_token_handler
from datetime import datetime as dt
from load_pred_pretrained_model import load_pred_pretrained_model


while True:
    if dt.now().hour == 16 and dt.now().minute == 35:
        token = api_token_handler()
        r = requests.get(url='http://192.168.115.10:8083/api/FlightLoadEstimate/GetAllFutureFlightsLoad',
                         headers={'Authorization': f'Bearer {token}'})
        load_pred_pretrained_model(json.loads(r.text)['getAllFutureFlightsLoadResponseItemViewModels'])
