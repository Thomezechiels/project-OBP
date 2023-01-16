import os
import pathlib

import dash
from dash import dcc, no_update, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_daq as daq
import plotly.express as px

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
from simulation.request import Request
from simulation.servers import ServerNetwork

global config
global state
global stop_sim

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Manufacturing SPC Dashboard"
server = app.server
app.config["suppress_callback_exceptions"] = True

suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_count = "_count"
suffix_ooc_n = "_OOC_number"
suffix_ooc_g = "_OOC_graph"
suffix_indicator = "_indicator"

def initState():
    return {
        'running_servers_prev': 0,
        'total_server_callback_count': 0,
        'stop_sim': True,
        'step': 0,
        'network': {
            'num_servers': 0,
            'servers_capacity': 0,
            'servers': {},
            'inactive_servers': {},
            'servers_used': [],
            
        }
    }

state = initState()

def generateRequest(arrival_prob):
    if np.random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        return Request(type, size, config['max_wait'])
    else:
        return False

def run_simulation(state_update = 200, live_mode = False):
    global state
    state['stop_sim'] = False
    serverNetwork = ServerNetwork(5, config['max_processes'])
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    while (t < end and not state['stop_sim']):
        arrival_prob = config['arrival_rates'][math.floor(t / steps)]
        if (t / steps).is_integer():
            serverNetwork.evaluate()
        request = generateRequest(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t, state_update)
        if live_mode:
            time.sleep(0.01)
        if t % state_update == 0:
            state['network'] = serverNetwork.outputState()
            state['step'] = t 
        t += 1
    
    if t >= steps:
        print("Simulation completed")
    return serverNetwork.outputState()

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

def build_tab_1():
    return (
        # Manually select metrics
        html.Div(
            id="set-specs-intro-container",
            children=html.P(
                "Use historical control limits to establish a benchmark, or set new values."
            ),
        ),
    )

def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)

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
                    generate_section_banner("Network Control Metrics Summary"),
                    html.Div(
                        id="metric-div",
                        children=[
                            generate_metric_list_header(),
                            html.Div(
                                id="metric-rows",
                                children=generate_server_statuses()
                            ),
                        ],
                    ),
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
        {"id": "m_header_6", "children": "Available"},
    )

def generate_server_statuses():
    ret = []
    global state
    servers = state['network']['servers']
    if servers:
        for idx, server in servers.items():
            ret.append(generate_metric_row_helper(str(idx), server['capacity'], server['active'], server['num_running_requests'], server['performance_history']))
    return ret

def generate_metric_row_helper(index, capacity, active, num_running_requests, performance_history):
    div_id = index + suffix_row
    button_id = index + suffix_button_id
    sparkline_graph_id = index + suffix_sparkline_graph
    count_id = index + suffix_count
    ooc_percentage_id = index + suffix_ooc_n
    ooc_graph_id = index + suffix_ooc_g
    indicator_id = index + suffix_indicator

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
                color={
                    "ranges": {
                        "#92e0d3": [0, 3],
                        "#f4d44d ": [3, 7],
                        "#f45060": [7, 10],
                    }
                },
                showCurrentValue=False,
                max=10,
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
            # html.Div(
            #     id="card-2",
            #     children=[
            #         html.P("Time to completion"),
            #         daq.Gauge(
            #             id="progress-gauge",
            #             max=max_length * 2,
            #             min=0,
            #             showCurrentValue=True,  # default size 200 pixel
            #         ),
            #     ],
            # ),
            html.Div(
                id="utility-card",
                children=[
                    html.Button(
                        "Start simulation", 
                        id="value-setter-set-btn",
                        n_clicks=0
                    ),
                    html.Div(
                        id="running-text",
                        children="Not running"
                    ),
                ],
            ),
        ],
    )

def update_sparkline(index, xaxis):
    global state
    servers = state['network']['servers']
    y_array = servers[index]["performance_history"][-1]
    x_array = (len(servers[index]["performance_history"])-1)*100
    if 'range' in xaxis and xaxis['range'][-1] < x_array: 
        print(index, x_array, y_array, xaxis['range'])
        return (dict(x=[[x_array]], y=[[y_array]]), [0])
    return None


def update_count(interval, index):
    if interval == 0:
        return "0", "0.00%", 0.00001, "#92e0d3"

    if interval > 0:
        global state
        server = state['network']['servers'][index]
        ooc_count = server['num_running_requests']
        queue_size = server['size_queue']
        ooc_percentage_f = (ooc_count / 10) * 100
        ooc_percentage_str = "%.2f" % ooc_percentage_f + "%"

        # Set maximum ooc to 10 for better grad bar display
        if ooc_count > 10:
            ooc_count = 10

        if ooc_count == 0.0:
            ooc_grad_val = 0.00001
        else:
            ooc_grad_val = float(ooc_count)

        # Set indicator theme according to threshold 5%
        if 0 <= ooc_grad_val <= 5:
            color = "#92e0d3"
        elif 5 < ooc_grad_val < 7:
            color = "#f4d44d"
        else:
            color = "#FF0000"

    return queue_size, ooc_percentage_str, ooc_grad_val, color


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval = 2 * 1000,
            n_intervals=0
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),
    ],
)
    
@app.callback(
    [Output("app-content", "children")],
    [Input("app-tabs", "value")],
)
def render_tab_content(tab_switch):
    if tab_switch == "tab1":
        return build_tab_1()
    return (
        html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=build_top_panel(),
                ),
            ],
        ),
    )

@app.callback(
    Output("n-interval-stage", "data"),
    [Input("app-tabs", "value")],
    [
        State("interval-component", "n_intervals"),
        State("interval-component", "disabled"),
        State("n-interval-stage", "data"),
    ],
)
def update_interval_state(tab_switch, cur_interval, disabled, cur_stage):
    if disabled:
        return cur_interval

    if tab_switch == "tab1":
        return cur_interval
    return cur_stage

# Callbacks for stopping interval update
# @app.callback(
#     [Output("interval-component", "disabled"), Output("value-setter-set-btn", "buttonText")],
#     [Input("stop-button", "n_clicks")],
#     [State("interval-component", "disabled")],
# )
# def stop_production(n_clicks, current):
#     if n_clicks % 2 == 0:
#         return True, "start"
#     return not current, "stop" if current else "start"

# ======= update status servers =========
@app.callback(
    output=[Output("graphs-container", "children")],
    inputs=[Input("interval-component", "n_intervals")],
)
def update_status_servers(interval):
    global state
    if not state['stop_sim']:
        running_prev = state['running_servers_prev']
        running_servers = state['network']['num_servers']
        if not running_prev == running_servers:
            total_server_callbacks = state['total_server_callback_count']
            state['running_servers_prev'] = running_servers
            if running_servers > total_server_callbacks:
                for index in range(total_server_callbacks, running_servers - total_server_callbacks):
                    state['total_server_callback_count'] = index
            return build_top_panel()
    return no_update

# Update LED clock
@app.callback(
    [Output('time-led', 'value')],
    [Input('interval-component', 'n_intervals')],
)
def update_clock(interval):
    global state
    d_hours = state['step'] / config['steps']
    hour = config['start_time'] + math.floor(d_hours)
    minutes = math.floor((d_hours % 1) * 60)
    time_val = ('0' if hour < 10 else '') + str(hour) + ':' + ('0' if minutes < 10 else '') + str(minutes)
    return [time_val]

# decorator for list of output
def create_callback(index):
    def callback(interval, figure):
        if not index in state['network']['servers']:
            return no_update
        queue, ooc_n, ooc_g_value, indicator = update_count(
            interval, index
        )
        # print(relayoutData)
        spark_line_data = update_sparkline(index, figure['layout']['xaxis'])
        return queue, spark_line_data, ooc_n, ooc_g_value, indicator

    return callback


for index in range(0, 100):
    update_index_row_function = create_callback(index)
    app.callback(
        output=[
            Output(str(index) + suffix_count, "children"),
            Output(str(index) + suffix_sparkline_graph, "extendData"),
            Output(str(index) + suffix_ooc_n, "children"),
            Output(str(index) + suffix_ooc_g, "value"),
            Output(str(index) + suffix_indicator, "color"),
        ],
        inputs=[Input("interval-component", "n_intervals"), Input(str(index) + suffix_sparkline_graph, 'figure')],
    )(update_index_row_function)

@app.callback(
    # Output("running-text", "children"),
    [ Output("value-setter-set-btn", "children"), Output("running-text", "children")],
    [Input("value-setter-set-btn", "n_clicks")],
)
def run_sim(n_clicks):
    if n_clicks % 2 == 1:
        live = True
        Thread(target=run_simulation, args=(200, live, )).start()
        return ["Stop simulation", "Running simulation"]
    else:
        global state
        state = initState()
        print('Stopped simulation')
        return ["Start simulation", "Not running"]

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
            config = yaml.safe_load(stream)
            app.run_server(debug=True, port=8050)
        except yaml.YAMLError as exc:
            print(exc)

    