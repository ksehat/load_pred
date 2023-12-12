import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from waitress import serve
from flask import Flask, request

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Initialize a global variable for the DataFrame
df = None

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Dropdown(
        id='route-dropdown',
        options=[],
        value='All',
        style={'width': '50%'}
    ),
    dcc.Graph(
        id='line-plot',
        style={'height': '700px'}
    )
])


@app.callback(
    [Output('route-dropdown', 'options'),
     Output('line-plot', 'figure')],
    [Input('url', 'pathname'),
     Input('route-dropdown', 'value')]
)
def update_graph(pathname, selected_route):
    global df
    if pathname == '/api1':
        # Get data from API 1
        df = ...
    elif pathname == '/api2':
        # Get data from API 2
        df = ...

    dropdown_options = [{'label': i, 'value': i} for i in df['route'].unique()]

    if selected_route == 'All':
        fig = px.line(df, x='datetime', y='error', hover_data=['route'])
    else:
        fig = px.line(df[df['route'] == selected_route], x='datetime', y='error')

    return dropdown_options, fig


if __name__ == '__main__':
    app.run_server(debug=True)
