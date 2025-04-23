import dash
from dash import dcc, html
import pandas as pd
import joblib
import os

xgb_model = joblib.load('models/xgb_model.pkl')
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Customer Churn Dashboard", style= {'textAlign': 'center'}),
    html.Div([
        html.Label("Stock code"),
        dcc.Input(id = 'stock', type = 'text'),

        html.Label("Country"),
        dcc.Input(id = 'country', type = 'text'),

        html.Label("Month of Invoice"),
        dcc.Input(id='month', type='number', min=1, max=12, step=1, value=6),

        html.Label("Day of Invoice"),
        dcc.Input(id='day', type='number', min=1, max=31, step=1, value=15),

        html.Label("Price per Unit"),
        dcc.Input(id='price', type='number', step=0.01, value=10.0),

        html.Label("Total Quantity Purchased"),
        dcc.Input(id='quantity', type='number', step=1, value=1),

        html.Label("Total Price"),
        dcc.Input(id='total', type='number', step=1, value=1),

        html.Br(),
        html.Button("Predict Churn", id='predict-btn', n_clicks=0),

        html.Div(id='prediction-output')
    ], className = "form-box")
])
@app.callback(
    dash.Output('prediction-output', 'children'),
    dash.Input('predict-btn', 'n_clicks'),
    dash.State('stock', 'value'),
    dash.State('country', 'value'),
    dash.State('month', 'value'),
    dash.State('day', 'value'),
    dash.State('price', 'value'),
    dash.State('quantity', 'value'),
    dash.State('total', 'value'),
)
def predict_segment(n_clicks, stock, country, month, day, price, quantity, total):
    if n_clicks >0:
        
        input_df = pd.DataFrame([{
            "StockCode": stock,
            "Country": country,
            "InvoiceMonth": month, 
            "InvoiceDay": day, 
            "UnitPrice": price, 
            "Quantity": quantity,
            "TotalPrice": total
        }])
        prediction = xgb_model.predict(input_df)[0]
        return f"Customer Churn Predicton: {'Yes' if prediction == 1 else 'No'}"
    return ""

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug = True, host = '0.0.0.0', port = port)