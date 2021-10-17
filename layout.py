import dash
import dash_core_components as dcc
import dash_html_components as html






def tab_layout(name, fig_plot, slider_app, out_table):
    layout = html.Div([
    html.H3(name, style={'text-align':'center'}),
    html.Div([
        html.Div([slider_app, fig_plot 
        ],),
        html.Br(),
        html.Div([
    html.H4('Prediction for next 14 days', style={'text-align':'center'})]),
        html.Div([out_table]),
        html.Div([
            html.H6(id='page-3-content')
        ], className='eight columns'),
    ], className='twelve columns'),
], className='twelve columns')
    return layout


