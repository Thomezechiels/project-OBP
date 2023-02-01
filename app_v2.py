import os
import pathlib

import scipy.stats

import dash
import dash_bootstrap_components as dbc
from dash import dcc, no_update, html, dash_table, ctx
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import dash_daq as daq

import pandas as pd
from argparse import ArgumentParser
import os
import pathlib
import yaml

from threading import Thread
import math
import time
import random
import numpy as np
from server_network.request import Request
from server_network.servers import ServerNetwork

import locale
locale.setlocale(locale.LC_ALL, '')

global progress
global await_response

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
    title="Manufacturing SPC Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

server = app.server
app.config["suppress_callback_exceptions"] = True

sim_params = ['start_time', 'end_time', 'size_period', 'steps', 'arrival_rates', 'prob_small', 'mean_small', 'std_small', 'mean_large',
              'std_large', 'max_wait', 'cost_server', 'cost_fail', 'reward_small', 'reward_large', 'min_servers', 'max_servers', 'max_processes']

suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_count = "_count"
suffix_ooc_n = "_OOC_number"
suffix_ooc_g = "_OOC_graph"
suffix_indicator = "_indicator"


def init_state():
    return {
        'changed': False,
        'network': False,
        'step': 0,
    }
    
def multiply_matrix(A,B):
  global C
  if  A.shape[1] == B.shape[0]:
    rows = B.shape[1]
    cols = A.shape[0]
    C = np.zeros((A.shape[0],B.shape[1]),dtype = int)
    for row in range(rows): 
        for col in range(cols):
            for elt in range(len(B)):
              C[row, col] += A[row, elt] * B[elt, col]
    return C
  else:
    return np.array([[20]])


def generate_request(arrival_prob):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(
            loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False


def run_simulation_test():
    serverNetwork = ServerNetwork(5, config['max_processes'], config=config,
                                  routing_policy=config['routing_chosen'], load_balancer=config['algorithm_chosen'])
    steps = config['steps']
    t = 0
    total_periods = config['end_time'] - config['start_time']
    end = total_periods * steps
    arrival_prob = 0
    
    context_vector_options_max = np.matrix([[1, 1, 1, 1/5]])
    context_vector_options_min = np.matrix([[1/7, scipy.stats.norm(0.5, 0.2).pdf(1/24)/2, 1/4, 1]])
    weights = np.matrix([[4],[8],[3],[-2]])
    max = multiply_matrix(context_vector_options_max, weights)[0][0]
    min = multiply_matrix(context_vector_options_min, weights)[0][0]
    
    global progress
    while (t < end):
        period = t / steps
        if period.is_integer():
            progress = period / total_periods
            if t > 0:
                num_servers, profit = serverNetwork.get_profit_period(t=t)
                # serverNetwork.train_lb(num_servers, X_t, profit)
           
            hour_context = scipy.stats.norm(0.5, 0.2).pdf(((period + 1) % 24)/24)/2
            X_t = np.array([random.randint(1, 7)/7, hour_context, random.randint(1, 4)/4, random.randint(1, 5)/5])
            arrival_prob = ((multiply_matrix(np.asmatrix(X_t), weights)[0][0] - min)/(max-min))*0.8 + 0.1
           
            serverNetwork.evaluate(X_t, period)
            # time.sleep(0.1)
        request = generate_request(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1
        
    serverNetwork.final_update()
    progress = 1.0
    state['network'] = serverNetwork.outputStateHistory()


def run_simulation_live():
    global await_response
    global state
    global progress
    serverNetwork = ServerNetwork(5, config['max_processes'], config=config,
                                  routing_policy=config['routing_chosen'], load_balancer=config['algorithm_chosen'])
    state['network'] = serverNetwork.outputStateHistory()
    steps = config['steps']
    t = 0
    total_periods = config['end_time'] - config['start_time']
    end = total_periods * steps
    arrival_prob = 0
    context_vector_options_max = np.matrix([[1, 1, 1, 1/5]])
    context_vector_options_min = np.matrix([[1/7, scipy.stats.norm(0.5, 0.2).pdf(1/24)/2, 1/4, 1]])
    weights = np.matrix([[4],[8],[3],[-2]])
    max = multiply_matrix(context_vector_options_max, weights)[0][0]
    min = multiply_matrix(context_vector_options_min, weights)[0][0]
    
    while t < end:
        period = t / steps
            
        if period.is_integer():
            if t > 0:
                num_servers, profit = serverNetwork.get_profit_period(t=t)
                # serverNetwork.train_lb(num_servers, X_t, profit)
                serverNetwork.updateState()
                state['network'] = serverNetwork.outputStateHistory()
                state['step'] = t
            hour_context = scipy.stats.norm(0.5, 0.2).pdf(((period + 1) % 24)/24)/2
            X_t = np.array([random.randint(1, 7)/7, hour_context, random.randint(1, 4)/4, random.randint(1, 5)/5])
            arrival_prob = ((multiply_matrix(np.asmatrix(X_t), weights)[0][0] - min)/(max-min))*0.8 + 0.1
            
            progress = serverNetwork.evaluate_live(X_t, period)
            await_response = True
            state['changed'] = True
            while await_response:
                if isinstance(await_response, bool):
                    time.sleep(0.5)
                else:
                    serverNetwork.setNActiveServers(await_response)
                    await_response = False
        elif t % round(steps / 12) == 0:
            serverNetwork.updateState()
            state['network'] = serverNetwork.outputStateHistory()
            state['step'] = t
            time.sleep(1.5)
                    
        request = generate_request(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
            
        t += 1
    progress = 1.0
    state['network'] = serverNetwork.outputStateHistory()


def build_profit_table():
    build_data = []
    for i in range(len(state['network']['profit']['total'])):
        build_data.append({
            'hour': i,
            'servers': state['network']['servers_used'][i],
            'rewards': (state['network']['profit']['rewards_small'][i] + state['network']['profit']['rewards_large'][i]),
            'server_costs': state['network']['profit']['server_costs'][i],
            'fails': state['network']['profit']['fails'][i],
            'total': state['network']['profit']['total'][i],

        })

    return [
        html.Div(
            'Total profit:  € {:n},-'.format(
                sum(state['network']['profit']['total'])),
            id="total-profit",
        ),
        dash_table.DataTable(
            id='table',
            columns=[
                {"name": 'Hour', "id": 'hour'},
                {"name": 'Servers', "id": 'servers'},
                {"name": 'Rewards', "id": 'rewards'},
                {"name": 'Server costs', "id": 'server_costs'},
                {"name": 'Fail costs', "id": 'fails'},
                {"name": 'Profit', "id": 'total'},
            ],
            data=build_data,
            style_data=dict(backgroundColor="rgba(0,0,0,0)"),
            style_table={'margin-top': '0px'},
            style_as_list_view=True,
            style_cell={
                'padding': '5px',
                'textAlign': 'left'
            },
            style_header={
                'backgroundColor': 'rgba(0,0,0,0)',
                'fontWeight': 'bold',
                'border': '1px solid #fff',
                'color': '#fff'
            },
            editable=True,
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{total} > 0',
                    },
                    'backgroundColor': 'rgba(0, 128, 0, .6)',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{total} < 0',
                    },
                    'backgroundColor': 'rgba(139, 0, 0, 0.6)',
                    'color': 'white'
                },
            ],
            page_size=20
        )
    ]


def build_arrival_graph():
    fig = go.Figure(
        {
            "layout": {
                "uirevision": True,
                "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                "xaxis": dict(
                    title="Hour",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    zeroline=False,
                    showticklabels=True,
                    color="#fff",
                    tick0=1,
                    dtick=1,
                ),
                "yaxis": dict(
                    title="Arrivals",
                    color="#fff",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    showticklabels=True,
                    tickfont=dict(
                        size=14,
                    ),
                ),
                "font_color": "white",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                "colorway": ['#636efa', '#008000'],
            },
        }
    )

    x = list(range(0, len(state['network']['arrivals']['small'])))
    fig.add_bar(x=x, y=state['network']['arrivals']
                ['small'], name="Small requests")
    fig.add_bar(x=x, y=state['network']['arrivals']
                ['large'], name="Large requests")
    fig.update_layout(barmode="relative")
    fig.update_xaxes(
        range=[-0.7, len(state['network']['arrivals']['small'])-0.5])

    return dcc.Graph(
        style={"width": "50%", "height": "25vh", "padding": "5px", },
        config={
            "staticPlot": True,
            "editable": False,
            "displayModeBar": True,
        },
        figure=fig,
    )


def build_completion_graph():
    completed_small = [r / config['reward_small']
                       for r in state['network']['profit']['rewards_small'][1:]]
    completed_large = [r / config['reward_large']
                       for r in state['network']['profit']['rewards_large'][1:]]
    fails = [f / -config['cost_fail']
             for f in state['network']['profit']['fails'][1:]]

    fig = go.Figure(
        {
            "layout": {
                "uirevision": True,
                "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                "xaxis": dict(
                    title="Hour",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    zeroline=False,
                    showticklabels=True,
                    color="#fff",
                    tick0=1,
                    dtick=1,
                ),
                "yaxis": dict(
                    title="Requests processed",
                    color="#fff",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    showticklabels=True,
                    tickfont=dict(
                        size=14,
                    ),
                ),
                "font_color": "white",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                'colorway': ['#ef553b', '#636efa', '#008000']
            },
        }
    )

    x = list(range(0, len(state['network']['profit']['fails'])))
    fig.add_bar(x=x, y=fails, name="Requests failed")
    fig.add_bar(x=x, y=completed_small, name="Small requests completed")
    fig.add_bar(x=x, y=completed_large, name="Large requests completed")

    fig.update_layout(barmode="relative")
    fig.update_xaxes(
        range=[-0.7, len(state['network']['profit']['fails'])-0.5])

    return dcc.Graph(
        style={"width": "50%", "height": "25vh", "padding": "5px", },
        config={
            "staticPlot": True,
            "editable": False,
            "displayModeBar": False,
        },
        figure=fig,
    )


def build_server_profit_graph():
    arrival_rates = [round((state['network']['arrivals']['small'][i] + state['network']['arrivals']
                           ['large'][i]) / config['steps'], 2) for i in range(len(state['network']['arrivals']['small']))]
    optimal = []
    for p in arrival_rates:
        a = config['optimal_servers'][p] if p in config['optimal_servers'] else 0
        optimal.append(a)
    fig = go.Figure(
        {
            "data": [
                {
                    "y": state['network']['servers_used'],
                    "x": list(range(0, len(state['network']['servers_used']))),
                    "mode": "lines+markers",
                    "name": 'Servers per hour',
                    "line": {"color": "#f4d44d"},
                },
                {
                    "y": optimal[:-1],
                    "x": list(range(0, len(optimal))),
                    "mode": "lines+markers",
                    "name": 'Optimal servers per hour',
                    "line": {"color": "rgba(255,255,255,.4)"},
                }
            ],
            "layout": {
                "uirevision": True,
                "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                "xaxis": dict(
                    title="Hour",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    zeroline=False,
                    showticklabels=True,
                    color="#fff",
                ),
                "yaxis": dict(
                    title="Servers used",
                    color="#fff",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    showticklabels=True,
                    tickfont=dict(
                        size=14,
                    ),
                ),
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                "font_color": "white",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }
    )

    fig.update_xaxes(range=[0, len(state['network']['servers_used'])-1])
    fig.add_hline
    return dcc.Graph(
        style={"width": "100%", "height": "25vh", "padding": "5px", },
        config={
            "staticPlot": True,
            "editable": False,
            "displayModeBar": True,
        },
        figure=fig,
    )


def build_waiting_times_graph():
    fig = go.Figure(
        {
            "data": [
                {
                    "y": state['network']['waiting_times_small'],
                    "x": list(range(0, len(state['network']['waiting_times_small']))),
                    "mode": "lines+markers",
                    "name": 'Average waiting times small requests',
                    "line": {"color": "#636efa"},
                },
                {
                    "y": state['network']['waiting_times_large'],
                    "x": list(range(0, len(state['network']['waiting_times_large']))),
                    "mode": "lines+markers",
                    "name": 'Average waiting times large requests',
                    "line": {"color": "#008000"},
                },
                {
                    "y": [config['max_wait_small']] * len(state['network']['waiting_times_small']),
                    "x": list(range(0, len(state['network']['waiting_times_small']))),
                    "mode": "lines",
                    "name": 'Maximum wait small request',
                    "line": {"color": "rgba(244, 212, 77, 0.4)"},
                },
                {
                    "y": [config['max_wait_large']] * len(state['network']['waiting_times_large']),
                    "x": list(range(0, len(state['network']['waiting_times_large']))),
                    "mode": "lines",
                    "name": 'Maximum wait large request',
                    "line": {"color": "rgba(255, 0, 0,.2)"},
                }
            ],
            "layout": {
                "uirevision": True,
                "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                "xaxis": dict(
                    title="Hour",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    zeroline=False,
                    showticklabels=True,
                    color="#fff",
                ),
                "yaxis": dict(
                    title="Average waiting time (seconds)",
                    color="#fff",
                    showline=True,
                    showgrid=True,
                    zeroline=False,
                    gridcolor="#333",
                    showticklabels=True,
                    tickfont=dict(
                        size=14,
                    ),
                ),
                'legend': dict(
                    # entrywidth=0.35, # change it to 0.3
                    # entrywidthmode='fraction',
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                "font_color": "white",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }
    )

    fig.update_xaxes(range=[0, len(state['network']['waiting_times_small'])-1])
    # fig.add_hline(y=config['max_wait_small'], )
    return dcc.Graph(
        style={"width": "100%", "height": "25vh", "padding": "5px", },
        config={
            "staticPlot": True,
            "editable": False,
            "displayModeBar": True,
        },
        figure=fig,
    )
    
def build_arrival_graph_live():
    fig = go.Figure(
        {
            "data": [
                {
                    "y": [0],
                    "x": [0],
                    "mode": "lines+markers",
                    "name": 'Arrivals',
                    "line": {"color": "#fff"},
                },
            ],
            "layout": {
                "uirevision": True,
                "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                "xaxis": dict(
                    title="Hour",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    zeroline=False,
                    showticklabels=True,
                    color="#fff",
                    range=[0,5],
                ),
                "yaxis": dict(
                    title="Number of arrivals",
                    color="#fff",
                    showline=True,
                    showgrid=True,
                    gridcolor="#333",
                    showticklabels=True,
                    range=[0,250],
                    tickfont=dict(
                        size=14,
                    ),
                ),
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                "font_color": "white",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }
    )

    return dcc.Graph(
        id="arrival-graph-live",
        style={"width": "100%", "height": "25vh", "padding": "5px", },
        config={
            "staticPlot": True,
            "editable": False,
            "displayModeBar": True,
        },
        figure=fig,
    )



def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Web server network control dashboard"),
                    html.H6("Network Performance and Control"),
                ],
            ),
            html.Button(
                'Change mode',
                id="mode-change-btn",
                n_clicks=0,
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.Div(
                        html.Img(
                            id="logo", src=app.get_asset_url("VU-logo.png"))
                    ),
                ],
            ),
        ],
    )


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="Parameter settings",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Network Status Dashboard",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def create_callback_param(param):
    def callback(value):
        config[param] = value
        return no_update
    return callback


for param in sim_params:
    update_param_row_function = create_callback_param(param)
    app.callback(
        output=[Output("sim_input_" + param, "value")],
        inputs=[Input("sim_input_" + param, "value")]
    )(update_param_row_function)


@app.callback(
    Output('routing-select-dropdown', 'value'),
    # Output('current-setup-routing', 'children'),
    Input('routing-select-dropdown', 'value')
)
def list_callback(value):
    config['routing_chosen'] = value
    return no_update


@app.callback(
    Output('algorithm-select-dropdown', 'value'),
    # Output('current-setup-algo', 'children'),
    Input('algorithm-select-dropdown', 'value')
)
def list_callback(value):
    config['algorithm_chosen'] = value
    return no_update


def generate_simulation_settings():
    ret = []
    black_list = ['algorithm_chosen', 'routing_chosen', 'optimal_servers']
    for input, value in config.items():
        if not isinstance(value, list) and not input in black_list:
            ret.append(
                html.Div(
                    id="sim_input_wrap_" + input,
                    children=(
                        daq.NumericInput(id=("sim_input_" + input), label=input +
                                         ":", className="setting-input", value=value, max=9999999)
                    )
                )
            )
    return ret


def build_tab_1():
    return (
        html.Div(
            id="settings-container",
            children=[
                # Manually select metrics
                html.Div(
                    id="set-specs-intro-container",
                    # className='twelve columns',
                    children=html.P(
                        "Select the routing policy and reward prediction algorithm to test its performance"
                    ),
                ),
                html.Div(
                    id="settings-menu",
                    children=[
                        html.Div(
                            id="metric-select-menu",
                            # className='five columns',
                            children=[
                                html.Div(
                                    id="routing-selection-wrap",
                                    children=[
                                        html.Label(
                                            id="routing-select-title", children="Select Routing Policy"),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id="routing-select-dropdown",
                                            options=list(
                                                {"label": alg, "value": alg} for alg in config['routing_policies']
                                            ),
                                            value=config['routing_chosen'],
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="algorithm-selection-wrap",
                                    children=[
                                        html.Label(
                                            id="algorithm-select-title", children="Select Algorithm"),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id="algorithm-select-dropdown",
                                            options=list(
                                                {"label": alg, "value": alg} for alg in config['algorithms']
                                            ),
                                            value=config['algorithm_chosen'],
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="simulation-settings-title",
                                    # className='twelve columns',
                                    children=html.P(
                                        "Settings simulation:"
                                    ),
                                ),
                                html.Div(
                                    id="simulation-settings",
                                    children=generate_simulation_settings()
                                )
                            ],
                        ),
                    ],
                ),
            ]
        ),
    )


def generate_server_statuses():
    ret = []
    if state['network']:
        servers = state['network']['servers']
        if servers:
            for idx, server in servers.items():
                ret.append(generate_metric_row_helper(
                    str(idx), server['num_running_requests']))
    return ret


def generate_metric_row_helper(index, performance_history):
    div_id = index + suffix_row
    button_id = index + suffix_button_id
    sparkline_graph_id = index + suffix_sparkline_graph
    count_id = index + suffix_count
    ooc_percentage_id = index + suffix_ooc_n
    ooc_graph_id = index + suffix_ooc_g
    indicator_id = index + suffix_indicator

    ranges = np.array_split(np.arange(config['max_processes'] + 1), 3)
    color = {
        'ranges': {
            "#92e0d3": [ranges[0][0], ranges[0][-1]],
            "#f4d44d": [ranges[1][0] - 1, ranges[1][-1]],
            "#f45060": [ranges[2][0] - 1, ranges[2][-1]],
        }
    }

    return generate_metric_row(
        str(int(div_id) + 1),
        None,
        {
            "id": index,
            "className": "metric-row-button-text",
            "children": html.Button(
                id=button_id,
                className="metric-row-button",
                children=index,
                title="Click to visualize live performance chart",
                n_clicks=0,
            ),
        },
        {"id": count_id, "children": "0"},
        {
            "id": index + "_sparkline",
            "children": dcc.Graph(
                id=sparkline_graph_id,
                style={"width": "100%", "height": "95%"},
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure=go.Figure(
                    {
                        "data": [
                            {
                                "y": performance_history,
                                "x": np.linspace(0, len(performance_history)*100, len(performance_history) + 1).tolist() if len(performance_history) > 1 else [0],
                                "mode": "lines+markers",
                                "name": index,
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    }
                ).update_layout(uirevision='constant'),
            ),
        },
        {"id": ooc_percentage_id, "children": "0.00%"},
        {
            "id": ooc_graph_id + "_container",
            "children": daq.GraduatedBar(
                id=ooc_graph_id,
                color=color,
                showCurrentValue=False,
                max=config['max_processes'],
                value=0,
            ),
        },
        {
            "id": index + "_pf",
            "children": daq.Indicator(
                id=indicator_id, value=True, color="#91dfd2", size=12
            ),
        },
    )


def generate_metric_row(id, style, col1, col2, col3, col4, col5, col6):
    if style is None:
        style = {"height": "8rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one column",
                style={"minWidth": "75px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center", "minWidth": "75px"},
                className="one column",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"height": "100%"},
                className="four columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"],
                style={},
                className="one column",
                children=col4["children"],
            ),
            html.Div(
                id=col5["id"],
                style={"height": "100%", "margin-top": "5rem"},
                className="three columns",
                children=col5["children"],
            ),
            html.Div(
                id=col6["id"],
                style={"display": "flex", "justifyContent": "center"},
                className="one column",
                children=col6["children"],
            ),
        ],
    )


def generate_section_banner(title, extra=None):
    return html.Div(className="section-banner", children=[title, extra])


def build_top_panel(live=False):
    children = None
    if live:
        children = [
            generate_section_banner("Network Control Metrics Summary",
                                    html.Div(
                                        id="info-number-active-servers",
                                        children=[
                                            "Using ",
                                            html.Span(
                                                id="num-used-servers",
                                                children="0"
                                            ),
                                            "/",
                                            str(config['max_servers']),
                                            " of available servers"
                                        ]
                                    ),
                                    ),
            html.Div(
                id="metric-div",
                children=[
                    generate_metric_list_header(),
                    html.Div(
                        id="metric-rows",
                        children=generate_server_statuses(),
                    ),
                ],
            ),
            html.Div(
                id="arrivals-live-wrap",
                children=[build_arrival_graph_live()]
            )
        ]
    elif state['network']:
        children = [
            generate_section_banner("Network Control Metrics Summary",
                                    html.P('Routing Policy: {} | Prediction Algorithm: {}'.format(
                                        config['routing_chosen'], config['algorithm_chosen']))
                                    ),
            html.Div(
                id='server-profit-graph',
                children=[
                    build_server_profit_graph(),
                    build_completion_graph(),
                    build_arrival_graph(),
                    build_waiting_times_graph(),
                ],
            ),
        ]

    return html.Div(
        id="metric-summary-session",
        className="eight columns",
        children=children,
    ),


# Build header
def generate_metric_list_header():
    return generate_metric_row(
        "metric_header",
        {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": "m_header_1", "children": html.Div("Server id")},
        {"id": "m_header_2", "children": html.Div("Queue size")},
        {"id": "m_header_3", "children": html.Div("Performance History")},
        {"id": "m_header_4", "children": html.Div("Load%")},
        {"id": "m_header_5", "children": html.Div("%Load")},
        {"id": "m_header_6", "children": "Active"},
    )


def generate_metric_row_helper(index, performance_history):
    div_id = index + suffix_row
    button_id = index + suffix_button_id
    sparkline_graph_id = index + suffix_sparkline_graph
    count_id = index + suffix_count
    ooc_percentage_id = index + suffix_ooc_n
    ooc_graph_id = index + suffix_ooc_g
    indicator_id = index + suffix_indicator

    ranges = np.array_split(np.arange(config['max_processes'] + 1), 3)
    color = {
        'ranges': {
            "#92e0d3": [ranges[0][0], ranges[0][-1]],
            "#f4d44d": [ranges[1][0] - 1, ranges[1][-1]],
            "#f45060": [ranges[2][0] - 1, ranges[2][-1]],
        }
    }

    return generate_metric_row(
        div_id,
        None,
        {
            "id": index,
            "className": "metric-row-button-text",
            "children": html.Button(
                id=button_id,
                className="metric-row-button",
                children=index,
                title="Click to visualize live performance chart",
                n_clicks=0,
            ),
        },
        {"id": count_id, "children": "0"},
        {
            "id": index + "_sparkline",
            "children": dcc.Graph(
                id=sparkline_graph_id,
                style={"width": "100%", "height": "95%"},
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure=go.Figure(
                    {
                        "data": [
                            {
                                "y": performance_history,
                                "x": np.linspace(0, len(performance_history)*100, len(performance_history) + 1).tolist() if len(performance_history) > 1 else [0],
                                "mode": "lines+markers",
                                "name": index,
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    }
                ).update_layout(uirevision='constant'),
            ),
        },
        {"id": ooc_percentage_id, "children": "0.00%"},
        {
            "id": ooc_graph_id + "_container",
            "children": daq.GraduatedBar(
                id=ooc_graph_id,
                color=color,
                showCurrentValue=False,
                max=config['max_processes'],
                value=0,
            ),
        },
        {
            "id": index + "_pf",
            "children": daq.Indicator(
                id=indicator_id, value=True, color="rgba(139, 0, 0, 1)", size=12
            ),
        },
    )


def generate_metric_row(id, style, col1, col2, col3, col4, col5, col6):
    if style is None:
        style = {"height": "8rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one column",
                style={"minWidth": "75px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center", "minWidth": "75px"},
                className="one column",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"height": "100%"},
                className="four columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"],
                style={},
                className="one column",
                children=col4["children"],
            ),
            html.Div(
                id=col5["id"],
                style={"height": "100%", "margin-top": "5rem"},
                className="three columns",
                children=col5["children"],
            ),
            html.Div(
                id=col6["id"],
                style={"display": "flex", "justifyContent": "center"},
                className="one column",
                children=col6["children"],
            ),
        ],
    )


def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="profit-table",
                children=build_profit_table() if state['network'] else None
            )
        ],
    )

def update_sparkline(index, xaxis):
    servers = state['network']['servers']
    y_array = servers[index]["num_running_requests"][-1]
    x_array = (len(servers[index]["num_running_requests"])-1)*100
    if 'range' in xaxis and xaxis['range'][-1] < x_array:
        return (dict(x=[[x_array]], y=[[y_array]]), [0])
    return None


def update_count(interval, index):
    if interval == 0:
        return "0", "0.00%", 0.00001, "#92e0d3"

    if interval > 0:
        server = state['network']['servers'][index]
        ooc_count = server['num_running_requests'][-1]
        queue_size = server['size_queue'][-1]
        ooc_percentage_f = (ooc_count / 10) * 100
        ooc_percentage_str = "%.2f" % ooc_percentage_f + "%"

        if ooc_count > 10:
            ooc_count = 10

        if ooc_count == 0.0:
            ooc_grad_val = 0.00001
        else:
            ooc_grad_val = float(ooc_count)

        ranges = np.array_split(np.arange(config['max_processes'] + 1), 3)
        # Set indicator theme according to threshold 5%
        if server['active'][-1]:
            color = "rgba(0, 128, 0, 1)"
        else:
            color = "rgba(139, 0, 0, 1)"

    return queue_size, ooc_percentage_str, ooc_grad_val, color


def buildSimRunning():
    return [
        dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
        html.Div(
            id='sim-running-screen',
            children=[
                html.P('Running simulation...'),
                dbc.Progress(id="progress", striped=True),
            ],
        ),
    ]


def build_decision_table():
    ret = []
    if isinstance(progress, dict):
        for i in progress:
            ret.append(
                html.Div(
                    id="choice-" + str(i),
                    children=[
                        # html.P(str(i), className="option"),
                        html.P('{}'.format(
                            progress[i]['servers']), className="num_servers"),
                        html.P('€ {:n}'.format(
                            progress[i]['profit']), className="profit"),
                        html.Button('Choose option', id="server-choice-" + \
                                    str(i), n_clicks=(int(progress[i]['servers']))-1),
                    ],
                ),
            )
    return ret


def buildDecisionPanel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="card-1",
                children=[
                    html.P("Current time"),
                    daq.LEDDisplay(
                        id="time-led",
                        value= ('0' if config['start_time'] < 10 else '') + str(config['start_time']) + ":00",
                        color="#92e0d3",
                        backgroundColor="#1e2130",
                        size=50,
                    ),
                ],
            ),
            html.Div(
                id="decision-table",
                children=[
                    html.Div(
                        id='table-header',
                        children=[
                            html.Div('Servers'),
                            html.Div('Projected profit'),
                            html.Div('Select option'),
                        ],
                    ),
                    html.Div(id='decision-table-body',
                             children=build_decision_table())
                ],
            )
        ],
    )


def build_live_sim_running():
    return [
        html.Div(
            id="live-sim-running-screen",
            children=[
                html.Div(
                    id="status-container",
                    children=[
                        buildDecisionPanel(),
                        html.Div(
                            id="graphs-container",
                            children=build_top_panel(True),
                        ),
                    ],
                )
            ],
        ),
        dcc.Interval("progress-interval-live", n_intervals=0,
                     disabled=False, interval=2000),
    ]


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval=1 * 1000,
            n_intervals=0
        ),
        dcc.Input(
            id="network-tab-state",
            type="hidden",
            value="start-screen",
        ),
        html.Button(
            id="sim-run-finished-btn",
            n_clicks=0,
            className="hidden",
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
                html.Div(
                    id="start-screen",
                    className="active",
                    children=[
                        html.Button(
                            children=[
                                html.I(className="bx bx-line-chart"),
                                'Start live mode (DSS)',
                            ],
                            id="live-mode-btn",
                            n_clicks=0,
                        ),
                        html.Button(
                            children=[
                                html.I(className="bx bxs-flask"),
                                'Algorithm testing mode',
                            ],
                            id="test-mode-btn",
                            n_clicks=0,
                        ),
                    ],
                ),
                html.Div(
                    id="test-mode-start-screen",
                    children=[
                        html.Img(src="assets/arrow.svg"),
                        html.P(
                            'Before you start the simulation, check the simulation settings to ensure the right routing policy and prediction algorithm are selected.'),
                        html.Button(
                            children=[
                                html.I(className="bx bx-play-circle"),
                                'Start simulation',
                                # html.P(
                                #     id="current-setup",
                                #     children=[
                                #         "With ",
                                #         html.P(id="current-setup-routing"),
                                #         "& ",
                                #         html.P(id="current-setup-algo"),
                                #     ],
                                # ),
                            ],
                            id="test-mode-start",
                            n_clicks=0,
                        ),
                        # html.Div(
                        #     id="current-setup",
                        #     children=
                        # )
                    ],
                ),
            ],
        ),

    ],
)

# Update LED clock


@app.callback(
    [Output('time-led', 'value')],
    [Input('progress-interval-live', 'n_intervals')],
)
def update_clock(interval):
    d_hours = state['step'] / config['steps']
    hour = config['start_time'] + math.floor(d_hours)
    minutes = math.floor((d_hours % 1) * 60)
    time_val = ('0' if hour < 10 else '') + str(hour) + ':' + \
        ('0' if minutes < 10 else '') + str(minutes)
    return [time_val]

# decorator for list of output


def create_callback(index):
    def callback(interval, figure):
        if not index in state['network']['servers']:
            return no_update
        queue, ooc_n, ooc_g_value, indicator = update_count(
            interval, index
        )
        spark_line_data = update_sparkline(index, figure['layout']['xaxis'])
        return queue, spark_line_data, ooc_n, ooc_g_value, indicator

    return callback


for index in range(0, 50):
    update_index_row_function = create_callback(index)
    app.callback(
        output=[
            Output(str(index) + suffix_count, "children"),
            Output(str(index) + suffix_sparkline_graph, "extendData"),
            Output(str(index) + suffix_ooc_n, "children"),
            Output(str(index) + suffix_ooc_g, "value"),
            Output(str(index) + suffix_indicator, "color"),
        ],
        inputs=[Input("progress-interval-live", "n_intervals"),
                Input(str(index) + suffix_sparkline_graph, 'figure')],
    )(update_index_row_function)

@app.callback(
    Output('arrival-graph-live', 'extendData'),
    Input('progress-interval-live', 'n_intervals')
)
def update_live_arrivals_graph(n):
    if len(state['network']['arrivals']['small']) > 1:
        arrivals = state['network']['arrivals']['small'][-2] + state['network']['arrivals']['large'][-2]
        x = round(state['step'] / config['steps'],2)
        arrivals = arrivals if arrivals > 0 else state['network']['arrivals']['small'][-3] + state['network']['arrivals']['large'][-3]
        # if 'range' in xaxis and xaxis['range'][-1] < x_array:
        return (dict(x=[[x]], y=[[arrivals]]), [0])
    return None

@app.callback(
    [Output("decision-table-body", "children"),
     Output("decision-table", "className"),
     Output('progress-interval-live', 'disabled')],
    [Input("progress-interval-live", "n_intervals"),
     Input("server-choice-1", "n_clicks"),
     Input("server-choice-2", "n_clicks"),
     Input("server-choice-3", "n_clicks")],
)
def update_decision_table(n_intervals, choice_1, choice_2, choice_3):
    if ctx.triggered_id == 'progress-interval-live' and state['changed']:
        state['changed'] = False
        return build_decision_table(), 'active', True
    elif ctx.triggered_id in ['server-choice-1', 'server-choice-2', 'server-choice-3']:
        for trigger in ctx.args_grouping:
            if trigger['id'] == ctx.triggered_id:
                global await_response
                await_response = int(trigger['value'])
                state['changed'] = False
                return no_update, 'inactive', False
    return no_update, no_update, no_update


@app.callback(
    [Output("progress", "value"), Output("progress", "label"),
     Output('sim-run-finished-btn', 'n_clicks')],
    [Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    progress_100 = round(progress * 100)
    finished = 1 if progress_100 >= 100 else no_update
    # finished = no_update
    return progress_100, f"{progress_100} %", finished


@app.callback(
    Output("app-content", "children"),
    Output('big-app-container', 'className'),
    Input("app-tabs", "value"),
    Input('network-tab-state', "value"),
)
def render_tab_content(tab_switch, network_tab_state):
    if tab_switch == "tab1":
        return build_tab_1(), 'none parameter-settings'
    else:
        overview_html = html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=build_top_panel(),
                ),
            ],
        )
        match network_tab_state:
            case "start-screen":
                return None, 'start-screen'
            case 'test-mode-start':
                return None, 'test-start-screen'
            case 'run-simulation':
                Thread(target=run_simulation_test).start()
                return buildSimRunning(), 'simulation-running'
            case 'run-simulation-live':
                Thread(target=run_simulation_live).start()
                time.sleep(0.2)
                return build_live_sim_running(), 'live-simulation-running'
            case "test-mode-finished":
                return overview_html, 'none'


@app.callback(
    [Output("network-tab-state", "value")],
    [Input("mode-change-btn", "n_clicks"),],
    [Input("test-mode-btn", "n_clicks")],
    [Input("live-mode-btn", "n_clicks")],
    [Input("test-mode-start", "n_clicks")],
    [Input("sim-run-finished-btn", "n_clicks")],
    prevent_initial_call=True
)
def render_network_tab(test_btn, live_btn, mode_btn, test_start_btn, sim_run_finished_btn):
    button_id = ctx.triggered_id if not None else 'None'
    if button_id == 'test-mode-btn':
        return 'test-mode-start',
    elif button_id == 'test-mode-start':
        return 'run-simulation',
    elif button_id == 'live-mode-btn':
        return 'run-simulation-live',
    elif button_id == 'mode-change-btn':
        return 'start-screen',
    elif button_id == 'sim-run-finished-btn':
        return 'test-mode-finished',


@app.callback(
    # Output("running-text", "children"),
    [Output("start-stop-simulation-btn", "children")],
    [Input("start-stop-simulation-btn", "n_clicks")],
)
def run_sim(n_clicks):
    if n_clicks % 2 == 1:
        return ["Stop simulation"]
    else:
        return ["Start simulation"]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config',
                        dest='config',
                        help='Select config to run the simulation with',
                        default='default.yaml', type=str)

    args = parser.parse_args()

    APP_PATH = str(pathlib.Path(__file__).parent.resolve())
    path = os.path.join(APP_PATH, os.path.join(args.config))

    with open(path, "r") as stream:
        try:
            progress = 0
            config = yaml.safe_load(stream)
            await_response = False
            config['algorithm_chosen'] = config['algorithms'][0]
            config['routing_chosen'] = config['routing_policies'][0]
            state = init_state()
            config['optimal_servers'] = pd.read_pickle(
                r'data/server_optimum.pickle')
            app.run_server(debug=True, port=8050)
        except yaml.YAMLError as exc:
            print(exc)
