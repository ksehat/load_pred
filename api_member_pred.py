import flask
from flask import Flask, request
import json
from waitress import serve
from load_pred import mean_load_pred
from load_pred2 import mean_load_pred2

app = Flask(__name__)
data_list = []


@app.route('/', methods=['POST', 'GET'])
def get_data():
    global data_list
    if (request.method == 'POST'):
        data_list.append(json.loads(request.data))
        return flask.Response(response=None)
    if (request.method == 'GET'):
        mean_load_pred2(data_list)
        data_list = []
        return flask.Response(response=None)


if __name__ == '__main__':
    serve(app, host='192.168.115.17', port='8080', threads=100)
