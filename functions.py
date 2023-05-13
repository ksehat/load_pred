import requests
import json
import os
from datetime import datetime as dt
def call_login_token():
    dict1 = {
        "username": "k.sehat",
        "password": "Ks@123456",
        "applicationType": 961,
        "iP": "1365"
    }
    r = requests.post(url='http://192.168.115.10:8081/api/Authentication/RequestToken',
                      json=dict1,
                      )
    token = json.loads(r.text)['token']
    expire_date = json.loads(r.text)['expires']
    return token, expire_date


def api_token_handler():
    if 'token_expire_date.txt' in os.listdir():
        with open('token_expire_date.txt', 'r') as f:
            te = f.read()
        expire_date = te.split('token:')[0]
        token = te.split('token:')[1]
        if dt.now() >= dt.strptime(expire_date, '%Y-%m-%d'):
            token, expire_date = call_login_token()
            expire_date = expire_date.split('T')[0]
            with open('token_expire_date.txt', 'w') as f:
                f.write(expire_date + 'token:' + token)
    else:
        token, expire_date = call_login_token()
        expire_date = expire_date.split('T')[0]
        with open('token_expire_date.txt', 'w') as f:
            f.write(expire_date + 'token:' + token)
    return token