import json
import requests

from functions import api_token_handler
from datetime import datetime as dt


while True:
    if dt.now().hour == 9 and dt.now().minute == 47:
        token = api_token_handler()
        r = requests.get(url='http://192.168.115.10:8081/api/Robots/GetAllFutureFlights',
                         headers={'Authorization': f'Bearer {token}'})
        if json.loads(r.content)['msg']:
            unsuccessful_pk = [int(x) for x in json.loads(r.text)['msg'].split(',')[1:]]
            for one_pk in unsuccessful_pk:
                r2 = requests.post(url='http://192.168.115.10:8081/api/Robots/GetAllFlightsWithSameCondition',
                                   json={'pkFlightInformation': one_pk},
                                   headers={'Authorization': f'Bearer {token}',
                                            'Content-type': 'application/json',
                                            })
            # TODO: this code should be continued.
        if not json.loads(r.content)['msg'] and json.loads(r.content)['success'] == True:
            requests.get(url='http://192.168.115.17:8080')
