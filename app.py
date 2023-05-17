import flask
from flask import Flask, request
import json
from waitress import serve
from load_pred_pretrained_model import load_pred_pretrained_model

# FLASK_APP = 'app.py'

app1 = Flask(__name__)


@app1.route('/', methods=['POST'])
def get_data():
    if (request.method == 'POST'):
        load_pred_pretrained_model(json.loads(request.data)['GetAllFutureFlightsResponseItemViewModels'])
        return flask.Response(response=None)


if __name__ == '__main__':
    serve(app1, host='192.168.115.17', port='8080', threads=100)
