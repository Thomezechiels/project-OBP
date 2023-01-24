from argparse import ArgumentParser
from pathlib import Path 
import yaml
import pickle
import json
from statistics import mean

import math
import random
import numpy as np
import pandas as pd

from server_network.request import Request
from server_network.servers import ServerNetwork

global config

def generateRequest(arrival_prob):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False    
    
def run_simulation():
    serverNetwork = ServerNetwork(5, config['max_processes'], config = config, routing_policy='round_robin', load_balancer='none')
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    arrival_prob = 0
    while (t < end):
        period = t / steps
        if (period).is_integer():
            if t > 0:
                num_servers, profit = serverNetwork.get_profit_period(t = t)
                serverNetwork.train_lb(num_servers, X_t, profit)
            X_t = np.random.normal(0, 0.6, 8)
            arrival_prob = 1 / (1 + math.exp(-X_t.sum()))
            serverNetwork.evaluate(X_t, period)
        request = generateRequest(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config',
                        dest='config',
                        help='Select config to run the simulation with',
                        default='default.yaml', type=str)

    args = parser.parse_args()
    filepath = Path(args.config)  
    filepath.parent.mkdir(parents=True, exist_ok=True)  

    with open(filepath, "r") as stream:
        try:
            random.seed(10)
            config = yaml.safe_load(stream)
            run_simulation()
        except yaml.YAMLError as exc:
            print(exc)