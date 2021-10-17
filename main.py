import requests
import requests_cache
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pystan
from prophet import Prophet
import plotly
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import os 

from layout import tab_layout
import json
with open('config.json','r') as f:
    api = json.load(f)


today = str(datetime.date.today())
requests_cache.install_cache('freecurrency_cache', backend='sqlite', expire_after=300)

def return_api():

    url = f"https://freecurrencyapi.net/api/v1/rates?base_currency=USD&date_from=2020-10-01&date_to={today}"

    headers = {
        'accept': "application/json",
        'content-type': "application/json",
        'apikey': api['API']
        }

    response = requests.request("GET", url, headers=headers)
    df = pd.DataFrame(response.json()['data'], index = None)
    df = df.transpose()
    df = df.reset_index()
    df = df.rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df["Date"],infer_datetime_format=True)
    return df
    
curr_dict = {'JPY':'Japanese Yen',  'GBP':'Great Britain Pound',
        'AUD':'Australian Dollar',  'CAD':'Canadian Dollar', 'CNY':'Chinese Yen',
       'HKD': 'Hongkong Dollar', 'INR':'Indian Rupee', 'SGD': 'Singapore Dollar'}
curren = ['JPY', 'GBP', 'AUD', 'CAD', 'CNY','HKD', 'INR', 'SGD']    
    
    
def currency_conv(df, splitval, currency):    
    if currency in curren:
        new_df = df[["Date",currency]][splitval:].dropna()
        new_df = new_df.rename(columns={'Date':'ds', currency:'y'})
        return new_df, df.shape[0]
    else:
        print("Enter valid currency")
        
def predictfn(df,period, frequency = 'D'):
    m = Prophet(changepoint_prior_scale=0.9,n_changepoints=30,
        changepoint_range = 0.9, daily_seasonality= True, 
        yearly_seasonality = False, weekly_seasonality = False)
    model = m.fit(df)
    seasonalities = model.seasonalities
    future_df = model.make_future_dataframe(periods=period, freq = frequency)
    forecast = model.predict(future_df)
    return forecast
    

def final_plot(df, forecast, currency, period):
    parameter = 'Daily Rate'


    title_d = curr_dict[currency]
    forecast = forecast.round(2)

    yhat = go.Scatter(x = forecast['ds'], y = forecast['yhat'], mode = 'lines', marker = {'color': '#E7B8B7'},
                      line = {'width': 3}, name = 'Forecast',)

    yhat_lower = go.Scatter(x = forecast['ds'], y = forecast['yhat_lower'], marker = {'color': 'rgba(178, 235, 241, 1)'},
                              showlegend = False, hoverinfo = 'none',)

    yhat_upper = go.Scatter(x = forecast['ds'], y = forecast['yhat_upper'], fill='tonexty',
                            fillcolor = 'rgb(223, 143, 164, 0.75)', name = 'Confidence', hoverinfo='none',mode = 'none')

    actual = go.Scatter(x = df['ds'], y = df['y'], mode = 'markers', marker = {'color': '#21130d','size': 4,
        'line': {'color': '#000000','width': .75}}, name = 'Actual')

    layout = go.Layout(yaxis = {'title': parameter, 'tickformat': format('y'), 'hoverformat': format('y')},
                hovermode = 'x', xaxis = { 'title': 'Date'}, margin = {'t': 20,'b': 50,'l': 60,'r': 10},
                  legend = {'bgcolor': 'rgba(0,0,0,0)'}, title={'text': f"{title_d}",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    data = [yhat_lower, yhat_upper, yhat, actual]
    
    return data, layout
        

    
def name_to_figure(currency, splitval2, period):
    
    
    filename = return_api()
    figure = go.Figure()
    
    
    if currency in curren:
        dist0_df = currency_conv(filename, splitval2, currency)[0]
        forecast0 = predictfn(dist0_df,period, frequency = 'D')
        data0, layout0 = final_plot(dist0_df, forecast0, currency, 14)
        fig = dict(data = data0, layout = layout0)
        pred_data = forecast0[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)
        pred_data['ds'] = pd.to_datetime(pred_data['ds'],errors='coerce').dt.date
        pred_data.rename(columns = {'ds':'Date','yhat':'Predicted', 'yhat_lower': 'Predicted Lower',
                                'yhat_upper':'Predicted Upper'}, inplace = True)   
        pred_data = pred_data.round(2)
        return fig, pred_data

myheading1 = 'Currency Rate Forecast'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                title= myheading1,
                update_title='Loading...',
                suppress_callback_exceptions=True)
server = app.server
                
colors = {"background": "#EEF4F7", "background_div": "white", 'text': '#DF8FA4'}
app.layout = html.Div(style={'marginTop': 0,'marginBottom': 10, 'backgroundColor': colors['background']}, children=[
    html.H1('Currency Rate Predictor', style={
            'textAlign': 'center',
            'color': colors['text'],
            'marginBottom': 8,'marginTop': 0
        }),
            html.P('Backtesting currency rates according to the currency value provided by the user.', style={
            'textAlign': 'center',
            'color': colors['text'],
            'marginBottom': 8,'marginTop': 0
        }),

html.Div([
    dcc.Tabs(id='tabs', value=curren[0], children=[
        dcc.Tab(label= curr_dict[curren[0]], value= curren[0], style={'textAlign': 'right'}),
        dcc.Tab(label= curr_dict[curren[1]], value= curren[1], style={'textAlign': 'right'}),
        dcc.Tab(label= curr_dict[curren[2]], value= curren[2], style={'textAlign': 'right'}),
        dcc.Tab(label= curr_dict[curren[3]], value= curren[3], style={'textAlign': 'right'}),
        dcc.Tab(label= curr_dict[curren[4]], value= curren[4], style={'textAlign': 'right'}),
        dcc.Tab(label= curr_dict[curren[5]], value= curren[5], style={'textAlign': 'right'}),
        dcc.Tab(label= curr_dict[curren[6]], value= curren[6], style={'textAlign': 'right'}),
        dcc.Tab(label= curr_dict[curren[7]], value= curren[7], style={'textAlign': 'right'}),
        ], vertical=True,parent_style={'float': 'left'}),
        html.Div(id='tabs-content', style={'width': '75%', 'float': 'left'})
    ]),
    html.Div([
    html.Br(),
        html.Div([html.A('Developed by Misha', href='https://github.com/mishaaggarwal15/currency_predictor', target='_blank')]),
        html.Br(),
      #  html.A("Homepage", href='/success'),
    ],style = {'textAlign': 'center'})
])

slider_app = html.Div([
    html.P('Set initial value', className = 'fix_label', style = {'text-align': 'center', 'color': 'Grey'}),
    dcc.Slider(
        id='slider_app',
        min= 0,
        max= 140,
        step= 5,
        value= 0,
        marks={i: str(i) for i in range(0, 140, 20)}
    ),
   html.Div(id='updatemode-output-container', style={'margin-top': 10, 'margin-bottom': 10})
])

@app.callback(
    dash.dependencies.Output('updatemode-output-container', 'children'),
    [dash.dependencies.Input('slider_app', 'value')])
def slider_output(value):
    return f'Selected Value: {int(value)}'


fig_plot = html.Div(id='fig_plot')
out_table = html.Div(id='out_table')
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == curren[0]:
        return tab_layout(curr_dict[curren[0]], fig_plot, slider_app, out_table)
        
    elif tab == curren[1]:
        return tab_layout(curr_dict[curren[1]], fig_plot, slider_app, out_table)
        
    elif tab == curren[2]:
        return tab_layout(curr_dict[curren[2]], fig_plot, slider_app, out_table)
        
    elif tab == curren[3]:
        return tab_layout(curr_dict[curren[3]], fig_plot, slider_app, out_table)
        
    elif tab == curren[4]:
        return tab_layout(curr_dict[curren[4]], fig_plot, slider_app, out_table)
        
    elif tab == curren[5]:
        return tab_layout(curr_dict[curren[5]], fig_plot, slider_app, out_table)
        
    elif tab == curren[6]:
        return tab_layout(curr_dict[curren[6]], fig_plot, slider_app, out_table)
        
    elif tab == curren[7]:
        return tab_layout(curr_dict[curren[7]], fig_plot, slider_app, out_table)
        

fig_name = curren
# Tab 1 callback
@app.callback([Output('fig_plot', 'children'),
                Output('out_table', 'children')],
              [Input('tabs', 'value'),
              Input('slider_app', 'value')])
def update_output(fig_name, slider_app):
    fig, table = name_to_figure(fig_name, slider_app, 14)    
    columns =  [{"name": i, "id": i,} for i in (table.columns)]
    return dcc.Graph(figure=fig), dash_table.DataTable(data=table.to_dict('records'), columns= columns,style_cell = {
                'font_family': 'sans-serif',
                'font_size': '16px',
                'text_align': 'center'
            })


############ Deploy
if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(port = 3050)

