from argparse import ArgumentParser
import yaml
import math
import random
import numpy as np
from request import Request
from servers import ServerNetwork
import os
import pathlib

def generateRequest(arrival_prob, config):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False    

def run_simulation(config):
    serverNetwork = ServerNetwork(4, config['max_processes'])
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    while (t < end):
        arrival_prob = config['arrival_rates'][math.floor(t / steps)]
        if (t / steps).is_integer():
            serverNetwork.evaluate()
        request = generateRequest(arrival_prob, config)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1

    serverNetwork.listServers()
    serverNetwork.printRunningRequests(t, [0,1,2,3,4])
    print('profit:', serverNetwork.calculate_profit(config['reward_small'], config['reward_large'], config['cost_fail'], config['cost_server']))

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
            run_simulation(config)
        except yaml.YAMLError as exc:
            print(exc)