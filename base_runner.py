import json
import requests
from functions import api_token_handler
from datetime import datetime as dt

while True:
    if dt.now().hour == 9 and dt.now().minute == 5:
        token = api_token_handler()
        r = requests.get(url='http://192.168.115.10:8081/api/Robots/GetAllFutureFlights',
                         headers={'Authorization': f'Bearer {token}'})
