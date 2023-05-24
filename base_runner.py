import json
import requests
from functions import api_token_handler
from datetime import datetime as dt

while True:
    if dt.now().hour == 10 and dt.now().minute == 38:
        token = api_token_handler()
        r = requests.get(url='http://192.168.115.10:8083/api/FlightLoadEstimate/GetAllFutureFlights',
                         headers={'Authorization': f'Bearer {token}'})
