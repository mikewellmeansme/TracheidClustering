from typing import Tuple
import dash_bootstrap_components as dbc
import numpy as np
from dash import (
    Dash,
    dash_table,
    dcc,
    html,
    Input,
    Output,
    callback
)
import plotly.express as px
import pandas as pd
from application import Application
from scipy.stats import mstats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime


app_pine = Application(
        'Borgad', 'app/input/БОГРАД_PISY.xlsx', 
        ['pisy_01a', 'pisy_01b', 'pisy_02a', 'pisy_03a', 'pisy_07a', 'pisy_12b', 'pisy_14a'],
        'app/input/Pine_crn.csv',
        'app/input/MinClimate.csv',
        {'PDSI': 'app/input/Minusinsk_PDSI.csv', 'SPEI': 'app/input/Minusinsk_SPEI.csv'}
    )

app_cedar = Application(
    'Kedr', 'app/input/КЕДР 100 перевал NEW (температура).xlsx',
    ['PISI04', 'PISI10a', 'PiSi_01', 'PISI10', '3PiSi13a', 'PISI14a', 'PISI17b'],
    'app/input/Cedar_crn.csv',
    'app/input/MinClimate.csv',
    {'PDSI': 'app/input/Minusinsk_PDSI.csv', 'SPEI': 'app/input/Minusinsk_SPEI.csv'},
)
app_cedar.change_class_names({
    0:1,
    1:2,
    2:3,
    3:0,
})

days_df = app_cedar.daily_climate_matcher.climate
days_df = days_df[days_df['Year']==2000].reset_index(drop=True)


dash_app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def get_period(relayout_data: dict) -> Tuple[int, int, int, int]:
    if 'xaxis.range[0]' not in relayout_data:
        return None

    start = datetime.datetime.strptime(relayout_data['xaxis.range[0]'], '%Y-%m-%d %H:%M:%S.%f')
    end = datetime.datetime.strptime(relayout_data['xaxis.range[1]'], '%Y-%m-%d %H:%M:%S.%f')
    if start.year != end.year:
        raise Exception('SELECT DAYS FROM THE SAME YEAR')

    mon0 = start.month
    mon1 = end.month
    day0 = start.day
    day1 = end.day
    return mon0, day0, mon1, day1


@callback(
    Output('boxplot-pine', 'figure'), 
    [
        Input('kw', 'relayoutData'),
        Input('column-selection-radiobutton', 'value'),
        Input('moving-average', 'value')
    ]
)
def update_pine_boxplot(relayout_data, column, moving_avg_window):
    period = get_period(relayout_data)
    if not period:
        return {'data': []}
    mon0, day0, mon1, day1 = period
    totals = app_pine.daily_climate_matcher.__get_total_climate__(
        app_pine.clustered_objects,
        column,
        [0,1,2,3],
        mon0,
        mon1,
        day0,
        day1,
        moving_avg_window,
        datetime.datetime.strptime(relayout_data['xaxis.range[0]'], '%Y-%m-%d %H:%M:%S.%f').year == 2000
    )

    fig = go.Figure()
    classes = [1,2,3,4]
    for cl, yd in zip(classes, totals):
            fig.add_trace(go.Box(
                y=yd,
                name=cl,
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                marker_size=2,
                line_width=1)
            )
    fig.update_layout({'title': f'Pine {column} ({day0}-{mon0} : {day1}-{mon1})'})
    
    return fig


@callback(
    Output('boxplot-cedar', 'figure'),
    [
        Input('kw', 'relayoutData'),
        Input('column-selection-radiobutton', 'value'),
    ]
)
def update_cedar_boxplot(relayout_data, column):
    period = get_period(relayout_data)
    if not period:
        return {'data': []}
    mon0, day0, mon1, day1 = period
    totals = app_cedar.daily_climate_matcher.__get_total_climate__(
        app_cedar.clustered_objects,
        column,
        [0,1,2],
        mon0,
        mon1,
        day0,
        day1,
        21,
        datetime.datetime.strptime(relayout_data['xaxis.range[0]'], '%Y-%m-%d %H:%M:%S.%f').year == 2000
    )
    totals.append([np.nan])
    fig = go.Figure()
    classes = [1,2,3,4]
    for cl, yd in zip(classes, totals):
            fig.add_trace(go.Box(
                y=yd,
                name=cl,
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                marker_size=2,
                line_width=1)
            )
    fig.update_layout({'title': f'Cedar {column} ({day0}-{mon0} : {day1}-{mon1})'})

    return fig


@callback(
    Output('kw', 'figure'),
    [
        Input('class-selection-radiobutton', 'value'),
        Input('moving-average', 'value'),
        Input('start-year', 'value'),
        Input('end-year', 'value'),
    ]
)
def start_kw(class_selection, moving_avg_window, start_year, end_year):
    def kruskall_test(df: pd.DataFrame, index: str):
        df = df[df['Year'].between(start_year, end_year)]
        groups = df.groupby('Class').groups
        classes = set(df['Class']) if class_selection == 'All classes' else [0,1,2]
        s, p = mstats.kruskalwallis(
                    *[list(df.loc[groups[j], index]) for j in classes]
                )
        return s, p
    
    cedar_df = app_cedar.daily_climate_matcher.climate.get_full_daily_comparison(app_cedar.clustered_objects, kruskall_test, moving_avg_window=moving_avg_window)
    pine_df = app_pine.daily_climate_matcher.climate.get_full_daily_comparison(app_pine.clustered_objects, kruskall_test, moving_avg_window=moving_avg_window)

    x = [datetime.date(2000, 1, 1) + datetime.timedelta(i) for i in range(366+366)]

    #x = list(range(1, len(pine_df) + 1))
    #prev_x = list(range(-len(pine_df) + 1, 1))
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02
    )
    fig.add_trace(go.Scatter(x=x, y=list(pine_df['Stat Temp prev']) + list(pine_df['Stat Temp']), name='KW Temp Pine', line={'color':'red','width': 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=list(pine_df['Stat Prec prev']) + list(pine_df['Stat Prec']), name='KW Prec Pine', line={'color':'blue','width': 2}), row=1, col=1)
    
    fig.add_hline(y=11.43, line_color = 'black',line_width = 1, line_dash = 'dash', name='p=0.01', row=1, col=1)
    fig.add_hline(y=7.8, line_color = 'gray',line_width = 1, line_dash = 'dash', name='p=0.05', row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=list(cedar_df['Stat Temp prev']) + list(cedar_df['Stat Temp']), name='KW Temp Cedar', line={'color':'red','width': 2}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=list(cedar_df['Stat Prec prev']) + list(cedar_df['Stat Prec']), name='KW Prec Cedar', line={'color':'blue','width': 2}), row=2, col=1)

    fig.add_hline(y=9, line_color = 'black',line_width = 1, line_dash = 'dash', name='p=0.01', row=2, col=1)
    fig.add_hline(y=6, line_color = 'gray',line_width = 1, line_dash = 'dash', name='p=0.05', row=2, col=1)

    fig.add_vline(x=datetime.date(2001, 1, 1), line_width=3, line_dash="dot", line_color="black")

    
    return fig



dash_app.layout = html.Div(children=[
    dbc.Container([
        html.H1(children='Kruskal–Wallis statistics'),
        dcc.RadioItems(
                ['All classes', 'Only 1-2-3'],
                value='All classes',
                id='class-selection-radiobutton',
        ),
        dcc.Input(value=21, id="moving-average", type="number"),
        dcc.Input(value=1918, id="start-year", type="number"),
        dcc.Input(value=2020, id="end-year", type="number"),
        dcc.Graph(
            id='kw',
            figure ={'data': []}
        ),
        dcc.RadioItems(
                ['Temperature', 'Precipitation'],
                'Temperature',
                id='column-selection-radiobutton',
        ),
        dbc.Row(
                [
                    dbc.Col(dcc.Graph(
                        id='boxplot-pine',
                        figure={'data': []}
                    ), width=6),
                    dbc.Col(dcc.Graph(
                        id='boxplot-cedar',
                        figure={'data': []}
                    ), width=6)
                ]
        )
    ])
])

if __name__ == '__main__':
    dash_app.run_server(debug=True)