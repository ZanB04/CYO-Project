import dash
from dash import dcc, html
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os

kmeans = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Customer Segmentation Dashboard", style= {'textAlign': 'center'}),
    html.Div([
        html.Label("Invoice Month"),
        dcc.Input(id='month', type='number', min=1, max=12, step=1, value=6),

        html.Label("Invoice Day"),
        dcc.Input(id='day', type='number', min=1, max=31, step=1, value=15),

        html.Label("Unit Price"),
        dcc.Input(id='price', type='number', step=0.01, value=10.0),

        html.Label("Quantity"),
        dcc.Input(id='quantity', type='number', step=1, value=1),

        html.Br(),
        html.Button("Predict Segment", id='predict-btn', n_clicks=0),

        html.Div(id='prediction-output', style={"marginTop": "20px", "fontSize": "20px"})
    ], style={"width": "50%", "margin": "auto"}),
])
@app.callback(
    dash.Output('prediction-output', 'children'),
    dash.Input('predict-btn', 'n_clicks'),
    dash.State('month', 'value'),
    dash.State('day', 'value'),
    dash.State('price', 'value'),
    dash.State('quantity', 'value'),
)
def predict_segment(n_clicks, month, day, price, quantity):
    if n_clicks > 0:
        features = np.array([[month, day, price, quantity]])
        scaled = scaler.transform(features)
        cluster = kmeans.predict(scaled)[0]
        return f"Preicted Customer Segment: {cluster}"
    return ""
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug = True, host = '0.0.0.0', port = port)