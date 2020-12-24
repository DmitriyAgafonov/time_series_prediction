import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

# from smooth_func import moving_avarage, exp_smoothing, \
#     holt_smoothing, holt_predict, holt_winters_predict
import smooth_func
from sarimax_model import sarimax_prediction

import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------

BS = dbc.themes.CYBORG

app = dash.Dash(__name__, external_stylesheets=[BS])


# ------------import and set dataframes-------------------

data1 = pd.read_csv(r'data/final_data.csv')
df1 = data1.iloc[2000:5000, :]
df_time = df1["TIME"]


data2 = pd.read_csv(r'data\final_data_predict.csv', index_col=0)
df2 = smooth_func.exp_smoothing(data2, 0.4)

df2_train, df2_test = df2.iloc[:round(len(df2)*0.9), :], df2.iloc[round(len(df2)*0.9):, :]

df_beer = data2['BEER_PROD'].to_frame()
df_beer_train, df_beer_test = df_beer.iloc[:round(len(df_beer)*0.9), :], df_beer.iloc[round(len(df_beer)*0.9):, :]
df2_train_, df2_test_ = df_beer_train['BEER_PROD'].to_frame(), df_beer_test['BEER_PROD'].to_frame()
pred, df2_train_ = sarimax_prediction(df2_train_, df2_test_)

smooth_drop = ['Moving Average', 'Exponential Smoothing', 'Holt Smoothing']

#print(df1.head(5))

def plt3():
    trace0 = go.Scatter(
        x=df2_train_.index, y=df2_train_['BEER_PROD'], name= 'Beer train part', marker ={'color': 'white'} ##1f77b4
            )

    trace1 = go.Scatter(
            x=df2_train_.index, y= df2_train_['sarima_fitted'], name='Fitted part', marker ={'color': '#099632'} #ff7f0e
        )

    trace2 = go.Scatter(
            x=df2_test_.index, y=df2_test_['BEER_PROD'], name= 'Beer test part', marker ={'color': 'white'}
        )

    trace3 = go.Scatter(
            x=df2_test_.index, y=pred, name= 'Predicted part', marker ={'color': '#ff2e82'} ###0cf74f
        )

    data= [trace0, trace1, trace2, trace3]
        # data =[]
    layout = go.Layout(
            title="Beer Prediction Graph",
            height=700,
        )

    fig = go.Figure(data=data, layout=layout)



    fig.add_shape(type="line",
                  # xref="x",
                  # yref="paper",
                  x0=df2_test_.index[0],
                  y0=-15,
                  x1=df2_test_.index[0],
                  y1=230,
                  line=dict(color="red", width=1, ),
                  # fillcolor=fillcolor,
                  # layer=layer
                  )

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            template='plotly_dark'
    )
    return fig

# --------------------------------------------------------


card_content = [
    dbc.CardHeader(
        #html.H6('Smoothing Parameters')
        html.P('Smoothing Parameters', style={'font-size': '20px', 'margin': '0 0 0 0'})
    ),
    dbc.CardBody(
        [dbc.Row([

            dbc.Col(
            html.Div([dbc.Row([
                dbc.Col(html.P(
                    "Change Moving Average parameter:"

                ),)#width={'size': 3, 'offset': 1})
            ]),

            dbc.Row([
                dbc.Col(
                    dcc.Slider(
                        id='mov_aver_param',
                        min=10,
                        max=150,
                        step=None,
                        marks={
                            10: '10',
                            25: '25',
                            50: '50',
                            100: '100',
                            150: '150'
                        },
                        value=50
                            ))]),    #, width={'size': 3, 'offset': 1})])

                dbc.Row([
                    dbc.Col(html.P(
                        "Change Exponential Smoothing parameter:"
                    ), style={'margin': '5% 0 0 0'} )  # , width={'size': 3, 'offset': 1})
                ]),

            dbc.Row([
                dbc.Col(
                    dcc.Slider(
                            id='exp_smooth_param',
                            min=0.005,
                            max=0.2,
                            step=None,
                            marks={
                                0.005: '0.005',
                                0.02: '0.02',
                                0.06: '0.06',
                                0.1: '0.1',
                                0.2: '0.2'
                            },
                            value=0.02
                        ))]) #, width={'size': 3, 'offset': 1})])

                    ], ),

        width={'size': 4, 'offset': 1}),

        dbc.Col(
            html.Div([
                dbc.Row([
                    dbc.Col(html.P(
                         "Change Smoothing Level (Holt):"
                            ),)#width={'size': 3, 'offset': 1})
                        ]),

                dbc.Row([
                    dbc.Col(
                    dcc.Slider(
                            id='holt_smooth_level',
                            min=0,
                            max=0.35,
                            step=None,
                            marks={
                                0: '0',
                                #0.005: '0.005',
                                #0.01: '0.01',
                                0.02: '0.02',
                                0.05: '0.05',
                                0.09: '0.09',
                                0.15: '0.15',
                                0.2: '0.2',
                                0.25: '0.25',
                                0.3: '0.3',
                                0.35: '0.35'
                            },
                            value=0.02
                        ))
                        ]),

                dbc.Row([
                    dbc.Col(html.P(
                         "Change Smoothing Trend (Holt):"
                            ), style={'margin': '5% 0 0 0'} )
                        ]),

                dbc.Row([
                    dbc.Col(
                    dcc.Slider(
                            id='holt_trend_level',
                            min=0,
                            max=0.1,
                            step=None,
                            marks={
                                0: '0',
                                #0.0005: '0.0005',
                                #0.001: '0.001',
                                0.005: '0.005',
                                #0.007: '0.007',
                                #0.01: '0.01',
                                #0.015: '0.015',
                                0.02: '0.02',
                                0.03: '0.03',
                                0.05: '0.05',
                                0.08: '0.08',
                                0.1: '0.1',
                            },
                            value=0.001
                        ))
                        ]),

            ]), width={'size': 4, 'offset': 1})
                ])

        ], style={"background-color": "#111111"})
]



app.layout = html.Div([

    html.H1("Graph smoothing", style={'margin': '1% 0 1% 7.5%'}),
    html.Hr(style={'background-color': 'grey', 'size': '1px', 'width': '85%'}),

        html.H5("Eye Tracking Series Smoothing", style={'margin': '1.5% 0 1% 8.4%'}),

            dbc.Row([
                dbc.Col(dbc.Label('Eye tracking lines'),
                        width={'size': 3, 'offset': 2}),

                dbc.Col(dbc.Label('Smooth type'),
                        width={'size': 3, 'offset': 2}),
            ]),

            dbc.Row([
                dbc.Col(

                    dcc.Dropdown(id='feature_drop1',
                                 options=[
                                     {'label': 'Pitch', 'value': df1.columns[1]},
                                     {'label': 'Yaw', 'value': df1.columns[2]},
                                     {'label': 'Roll', 'value': df1.columns[3]}
                                 ],
                                 value=[df1.columns[1], df1.columns[2], df1.columns[3]],
                                 style = {"background-color": "#1A1A1A", 'margin': '0 0 5% 0', 'color': 'black'},
                                 multi=True

                                 ),
                    width={'size': 3, 'offset': 2}
                ),
                dbc.Col(

                    dcc.Dropdown(id='case_smooth_drop',
                                 options=[
                                     # {'label': 'none', 'value': 'none'},
                                     {'label': smooth_drop[0], 'value': smooth_drop[0]},
                                     {'label': smooth_drop[1], 'value': smooth_drop[1]},
                                     {'label': smooth_drop[2], 'value': smooth_drop[2]},
                                 ],
                                 value=[],
                                 style = {"background-color": "#1A1A1A", 'margin': '0 0 5% 0', 'color': 'black'},
                                 multi=True

                                 ),
                    width={'size': 3, 'offset': 2}
                ),

            ]),



        dbc.Row(
            [

                dbc.Col(dbc.Card(card_content, inverse=True),
                        width={'size': 10, 'offset': 2}, style={'margin': '1% 8.333% 1% 8.333%'})
            ]),





        dbc.Row([

            dbc.Col(dcc.Graph(id='graph1'),
                # width=10, lg={'size': 10, 'offset': 1}
                width={'size': 10, 'offset': 1}
                    )

        ]),

    dbc.Row([

        dbc.Col(dbc.Table.from_dataframe(df1.head(5), striped=True, bordered=True, hover=True),
                width={'size': 10, 'offset': 1}
                # width=10, lg={'size': 10, 'offset': 1}
                )

    ]),

    html.Hr(style={'background-color': 'grey', 'size': '1px', 'width': '83%'}),
    html.H5("Time Series Prediction", style={'margin': '1.5% 0 1% 8.4%'}),

        dbc.Row([
                dbc.Col(dbc.Label('Predict lines'),
                        width={'size': 3, 'offset': 2}),

                dbc.Col(dbc.Label('Prediction Method'),
                        width={'size': 3, 'offset': 2}),
            ]),

            dbc.Row([
                dbc.Col(

                    dcc.Dropdown(id='feature_drop2',
                                 options=[
                                     {'label': df2_train.columns[0], 'value': df2_train.columns[0]},
                                     {'label': df2_train.columns[1], 'value': df2_train.columns[1]},
                                     {'label': df2_train.columns[2], 'value': df2_train.columns[2]},
                                     {'label': df2_train.columns[3], 'value': df2_train.columns[3]}
                                 ],
                                 value=df2_train.columns[3],
                                 clearable=False,
                                 style={"background": "#1A1A1A", 'margin': '0 0 5% 0', 'color': 'black'},

                                 ),
                    width={'size': 3, 'offset': 2}
                ),
                dbc.Col(

                    dcc.Dropdown(id='predict_type_drop',
                                 options=[

                                     {'label': 'Holt Prediction', 'value': 'h_prediction'},
                                     {'label': 'Holt-Winters Prediction', 'value': 'hw_prediction'}

                                 ],
                                 value='h_prediction',
                                 clearable=False,
                                 style={"background-color": "#1A1A1A", 'margin': '0 0 5% 0', 'color': 'black'}

                                 ),
                    width={'size': 3, 'offset': 2}
                ),

            ]),

        dbc.Row([

            dbc.Col(dcc.Graph(id='graph2'),
                # width=10, lg={'size': 10, 'offset': 1}
                width={'size': 10, 'offset': 1}
                    )

        ]),

    dbc.Row([

        dbc.Col(dbc.Table.from_dataframe(df2.head(5), striped=True, bordered=True, hover=True),
                # width=10, lg={'size': 10, 'offset': 1}
                width={'size': 10, 'offset': 1}
                )

    ]),

    html.Hr(style={'background-color': 'grey', 'size': '1px', 'width': '83%'}),
    html.H5("Time Series Prediction SARIMAX", style={'margin': '1.5% 0 1% 8.4%'}),

    dbc.Row([

            dbc.Col(dcc.Graph(id='graph3', figure=plt3()),
                # width=10, lg={'size': 10, 'offset': 1}
                width={'size': 10, 'offset': 1}
                    )

        ]),


])




@app.callback(
    Output('graph1', 'figure'),
    [Input('feature_drop1', 'value'),
     Input('case_smooth_drop', 'value'),
     Input('mov_aver_param', 'value'),
     Input('exp_smooth_param', 'value'),
     Input('holt_smooth_level', 'value'),
     Input('holt_trend_level', 'value')])
def plt1(feature_drop1, case_smooth_drop, mov_aver_param, exp_smooth_param, holt_smooth_level, holt_trend_level):
    m_a_df = smooth_func.moving_avarage(df1, mov_aver_param)
    ex_s_df = smooth_func.exp_smoothing(df1, exp_smooth_param)
    h_s_df = smooth_func.holt_smoothing(df1, holt_smooth_level, holt_trend_level)

    dfs = [m_a_df, ex_s_df, h_s_df]
    dict_dfs = dict(zip(smooth_drop, dfs))


    data = []

    for i in feature_drop1:
        data.append(go.Scatter(
            x=df_time, y=df1[i], name=i
        ))

    for j in case_smooth_drop:
        for i in feature_drop1:
            data.append(go.Scatter(
                x=df_time, y=dict_dfs[j][i], name=i + ' ' + j
            ))

    layout = go.Layout(
                 title="Eye Tracking Parameters Graph",
                 height=670,
                    )


    fig = go.Figure(data=data, layout=layout)

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Deviation",
        template='plotly_dark')

    fig.update_xaxes(nticks=7,
                     tickangle = 90)


    return fig


@app.callback(
    Output('graph2', 'figure'),
    [Input('feature_drop2', 'value'),
     Input('predict_type_drop', 'value')])
def plt2(feature_drop2, predict_type_drop):
    if predict_type_drop == 'h_prediction':
        ts = smooth_func.holt_predict(df2_train, df2_test, feature_drop2)
    else:
        ts = smooth_func.holt_winters_predict(df2_train, df2_test, feature_drop2)

    trace0 = go.Scatter(
        x=df2_train.index, y=df2_train[feature_drop2], name=feature_drop2 +' '+ 'train part'
            )

    trace1 = go.Scatter(
        x=df2_test.index, y=df2_test[feature_drop2], name=feature_drop2 +' ' + 'test part'
    )

    trace2 = go.Scatter(
        x=df2_test.index, y=ts, name=feature_drop2 +' ' + 'predicted part'
    )

    data= [trace0, trace1, trace2]
    # data =[]
    layout = go.Layout(
        title="Min, Max Temperature, Electric and Beer Production Graph",
        height=700,
    )

    fig = go.Figure(data=data, layout=layout)


    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        template='plotly_dark')



    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
