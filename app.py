import flask
from flask import Flask, request
import json
from waitress import serve
from load_pred_ML import mean_load_pred
from load_pred import mean_load_pred

# FLASK_APP = 'app.py'

app1 = Flask(__name__)


@app1.route('/', methods=['POST'])
def get_data():
    data_list = []
    if (request.method == 'POST'):
        data_list.append(json.loads(request.data))
        mean_load_pred(data_list)
        return flask.Response(response=None)


if __name__ == '__main__':
    serve(app1, host='192.168.115.17', port='8080', threads=100)
