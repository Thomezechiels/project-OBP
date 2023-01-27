import os
import pathlib

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

global config
global state
global progress
global await_response

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Manufacturing SPC Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

server = app.server
app.config["suppress_callback_exceptions"] = True

sim_params = ['start_time', 'end_time', 'size_period', 'steps', 'arrival_rates', 'prob_small', 'mean_small', 'std_small', 'mean_large', 'std_large', 'max_wait', 'cost_server', 'cost_fail', 'reward_small', 'reward_large', 'min_servers', 'max_servers', 'max_processes']

suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_count = "_count"
suffix_ooc_n = "_OOC_number"
suffix_ooc_g = "_OOC_graph"
suffix_indicator = "_indicator"

def generateRequest(arrival_prob):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False    
    
def run_simulation_test():
    serverNetwork = ServerNetwork(5, config['max_processes'], config = config, routing_policy=config['routing_chosen'], load_balancer=config['algorithm_chosen'])
    steps = config['steps']
    t = 0
    total_periods = config['end_time'] - config['start_time']
    end = total_periods * steps
    arrival_prob = 0
    global progress
    while (t < end):
        period = t / steps
        if period.is_integer():
            progress = period / total_periods
            if t > 0:
                num_servers, profit = serverNetwork.get_profit_period(t = t)
                serverNetwork.train_lb(num_servers, X_t, profit)
            X_t = np.random.normal(0, 0.6, 8)
            arrival_prob = 1 / (1 + math.exp(-X_t.sum()))
            serverNetwork.evaluate(X_t, period)
            time.sleep(0.1)
        request = generateRequest(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1
    progress = 1.0
    global state
    state = serverNetwork.outputStateHistory()

def run_simulation_live():
    serverNetwork = ServerNetwork(5, config['max_processes'], config = config, routing_policy=config['routing_chosen'], load_balancer=config['algorithm_chosen'])
    steps = config['steps']
    t = 0
    total_periods = config['end_time'] - config['start_time']
    end = total_periods * steps
    arrival_prob = 0
    global await_response
    await_response = True
    global progress
    while (t < end):
        period = t / steps
        if period.is_integer():
            if t > 0:
                num_servers, profit = serverNetwork.get_profit_period(t = t)
                serverNetwork.train_lb(num_servers, X_t, profit)
            X_t = np.random.normal(0, 0.6, 8)
            arrival_prob = 1 / (1 + math.exp(-X_t.sum()))
            progress = serverNetwork.evaluate_live(X_t, period)
            # while(await_response):
            #     time.sleep(1)
            print(progress)
        request = generateRequest(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1
    progress = 1.0
    global state
    state = serverNetwork.outputStateHistory()

def buildProfitTable():
    build_data = []
    for i in range(len(state['profit']['total'])):
        build_data.append({
            'hour': i,
            'servers': state['servers_used'][i],
            'rewards': state['profit']['rewards'][i],
            'server_costs': state['profit']['server_costs'][i],
            'fails': state['profit']['fails'][i],
            'total': state['profit']['total'][i],
                                                                                        
        })

    return [
        html.Div(
            'Total profit:  € {:n},-'.format(sum(state['profit']['total'])),
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
            data= build_data,
            style_data=dict(backgroundColor="rgba(0,0,0,0)"),
            style_table={'margin-top': '0px'},
            style_as_list_view=True,
            style_cell={
                'padding': '5px',
                'textAlign':'left'
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
                    'backgroundColor': 'green',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{total} < 0',
                    },
                    'backgroundColor': '#8b0000',
                    'color': 'white'
                },
            ],
            page_size=20
        )
    ]

def buildArrivalGraph():
    fig = go.Figure(
                    {
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                title="Hour",
                                showline=True,
                                showgrid=True,
                                gridcolor = "#333",
                                zeroline=False,
                                showticklabels=True,
                                color = "#fff",
                            ),
                            "yaxis": dict(
                                title="Arrivals",
                                color = "#fff",
                                showline=True,
                                showgrid=True,
                                gridcolor = "#333",
                                showticklabels=True,
                                tickfont = dict(
                                    size = 14,
                                ),
                            ),
                            "font_color":"white",
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    }
                )
    
    x = list(range(0, len(state['arrivals']['small']) * 2))
    fig.add_bar(x=x, y=state['arrivals']['small'], name="Small requests")
    fig.add_bar(x=x, y=state['arrivals']['large'], name="Large requests")
    fig.update_layout(barmode="relative")

    return dcc.Graph(
            style={"width": "100%", "height": "30vh", "padding": "20px",},
            config={
                "staticPlot": True,
                "editable": False,
                "displayModeBar": True,
            },
            figure = fig,
    )

def buildServerProfitGraph():
    fig = go.Figure(
        {
            "data": [
                {
                    "y": state['servers_used'] * 2,
                    "x": list(range(0, len(state['servers_used']) * 2)),
                    "mode": "lines+markers",
                    "name": 'Servers per hour',
                    "line": {"color": "#f4d44d"},
                }
            ],
            "layout": {
                "uirevision": True,
                "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                "xaxis": dict(
                    title="Hour",
                    showline=True,
                    showgrid=True,
                    gridcolor = "#333",
                    zeroline=False,
                    showticklabels=True,
                    color = "#fff",
                ),
                "yaxis": dict(
                    title="Servers used",
                    color = "#fff",
                    showline=True,
                    showgrid=True,
                    gridcolor = "#333",
                    showticklabels=True,
                    tickfont = dict(
                        size = 14,
                    ),
                ),
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }
    )
    
    fig.update_xaxes(range=[0,len(state['servers_used'])*2-1])
    return dcc.Graph(
        style={"width": "100%", "height": "30vh", "padding": "20px",},
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
                            html.Img(id="logo", src=app.get_asset_url("VU-logo.png"))
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
    for input, value in config.items():
        if not isinstance(value, list):
            ret.append(
                html.Div(
                    id="sim_input_wrap_" + input,
                    children=(
                        daq.NumericInput(id=("sim_input_" + input), label=input+":", className="setting-input", value=value, max=9999999)
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
                                    children = [
                                        html.Label(id="routing-select-title", children="Select Routing Policy"),
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
                                    children = [
                                        html.Label(id="algorithm-select-title", children="Select Algorithm"),
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
                        # html.Div(
                        #     id="value-setter-menu",
                        #     # className='six columns',
                        #     children=[
                        #         html.Div(id="value-setter-panel"),
                        #         html.Br(),
                        #         html.Div(
                        #             id="button-div",
                        #             children=[
                        #                 html.Button("Update", id="value-setter-set-btn"),
                        #                 html.Button(
                        #                     "View current setup",
                        #                     id="value-setter-view-btn",
                        #                     n_clicks=0,
                        #                 ),
                        #             ],
                        #         ),
                        #         html.Div(
                        #             id="value-setter-view-output", className="output-datatable"
                        #         ),
                        #     ],
                        # ),
                    ],
                ),
            ]
        ), 
    )

def generate_section_banner(title, extra=None):
    return html.Div(className="section-banner", children=[title, extra])

def build_top_panel():
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("Network Control Metrics Summary",
                        # html.Div(
                        #     id="info-number-active-servers",
                        #     children=[
                        #         "Using ",
                        #         html.Span(
                        #             id="num-used-servers",
                        #             children="0"
                        #         ),
                        #         "/",
                        #         str(config['max_servers']),
                        #         " of available servers"
                        #     ]
                        # )
                    ),

                    html.Div(
                        id='server-profit-graph',
                        children = [
                            buildServerProfitGraph(),
                            buildArrivalGraph(),                      
                        ],
                    ),
                    # html.Div(
                    #     id="metric-div",
                    #     children=[
                    #         generate_metric_list_header(),
                    #         html.Div(
                    #             id="metric-rows",
                    #             children=generate_server_statuses()
                    #         ),
                    #     ],
                    # ),
                ],
            ),
        ],
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
        {"id": "m_header_6", "children": "Availability"},
    )

# def generate_server_statuses():
#     ret = []
#     servers = state['network']['servers']
#     if servers:
#         for idx, server in servers.items():
#             ret.append(generate_metric_row_helper(str(idx), server['capacity'], server['active'], server['num_running_requests'], server['performance_history']))
#     return ret

def generate_metric_row_helper(index, capacity, active, num_running_requests, performance_history):
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

def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="profit-table",
                children=buildProfitTable()
            )
        ],
    )

def update_sparkline(index, xaxis):
    servers = state['network']['servers']
    y_array = servers[index]["performance_history"][-1]
    x_array = (len(servers[index]["performance_history"])-1)*100
    if 'range' in xaxis and xaxis['range'][-1] < x_array: 
        return (dict(x=[[x_array]], y=[[y_array]]), [0])
    return None


def update_count(interval, index):
    if interval == 0:
        return "0", "0.00%", 0.00001, "#92e0d3"

    if interval > 0:
        server = state['network']['servers'][index]
        ooc_count = server['num_running_requests']
        queue_size = server['size_queue']
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
        if 0 <= ooc_grad_val <= ranges[0][-1]:
            color = "#92e0d3"
        elif ranges[0][-1] < ooc_grad_val < ranges[1][-1]:
            color = "#f4d44d"
        else:
            color = "#FF0000"

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


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval = 1 * 1000,
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
                            children= [
                                html.I(className="bx bx-line-chart"),
                                'Start live mode (DSS)',                            
                            ],
                            id="live-mode-btn",
                            n_clicks=0, 
                        ),
                        html.Button(
                            children= [
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
                        html.P('Before you start the simulation, check the simulation settings to ensure the right routing policy and prediction algorithm are selected.'),
                        html.Button(
                            children= [
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

@app.callback(
    [Output("progress", "value"), Output("progress", "label"), Output('sim-run-finished-btn', 'n_clicks')],
    [ Input("progress-interval", "n_intervals")],
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
                return buildSimRunning(),'simulation-running'
            case 'run-simulation-live':
                Thread(target=run_simulation_live).start()
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
    # print(button_id)
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
    [ Output("start-stop-simulation-btn", "children")],
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
            state = pd.read_pickle(r'data/network_history.pickle')
            app.run_server(debug=True, port=8050)
        except yaml.YAMLError as exc:
            print(exc)

    