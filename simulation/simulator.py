from argparse import ArgumentParser
import yaml
import math
import random
import numpy as np
import pandas as pd
from request import Request
from servers import ServerNetwork
import os
import pathlib
from pathlib import Path 

def generateRequest(arrival_prob, config):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False    

def run_simulation(config, data_generation = False):
    serverNetwork = ServerNetwork(5, config['max_processes'], routing_policy='round_robin')
    serverNetwork.setConfig(config)
    data_points = []
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    while (t < end + 1):
        if t < end:
            arrival_prob = config['arrival_rates'][math.floor(t / steps)]
        if (t / steps).is_integer():
            if data_generation and t > 0:
                arrival = config['arrival_rates'][math.floor((t -1)/ steps)]
                num_servers, profit, workload = serverNetwork.evaluate(t = t, data_generation = True)
                data_point = {
                    "action": num_servers,
                    "cost": -profit,
                    "probability": 0.1,
                    "feature_arrival": arrival,
                    "feature_workload": int(round(workload, -3)),
                }
                data_points.append(data_point)
            elif t == 0 and data_generation:
                num_servers, profit, workload = serverNetwork.evaluate(t = t, data_generation = True)
            elif not data_generation:
                # print('\nPeriod', (t/steps))
                serverNetwork.evaluate(t = t, arrivals=arrival_prob, use_lb = True)
                # serverNetwork.listServers()
        if t < end:
            request = generateRequest(arrival_prob, config)
            if (request and request.size > 0):
                serverNetwork.handleRequest(t, request)
            serverNetwork.update(t)
        t += 1

    if not data_generation:
        print(serverNetwork.calculate_profit())
    return data_points

def generateData(config):
    train_data = []
    for i in range(1000):
        data_points = run_simulation(config, True)
        train_data += data_points
        print('Finished run:', i)
    
    train_df = pd.DataFrame(train_data)
    # Add index to data frame
    train_df["index"] = range(1, len(train_df) + 1)
    train_df = train_df.set_index("index")

    filepath = Path('data/training.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    train_df.to_csv(filepath)
    

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
            random.seed(10)
            config = yaml.safe_load(stream)
            run_simulation(config)
            # generateData(config)
        except yaml.YAMLError as exc:
            print(exc)