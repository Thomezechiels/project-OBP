from argparse import ArgumentParser
import yaml
import math
import random
import numpy as np
from request import Request
from servers import ServerNetwork

global config

def generateRequest(arrival_prob):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        return Request(type, size)
    else:
        return False    

def run_simulation():
    serverNetwork = ServerNetwork(5, config['max_processes'])
    steps = config['steps']
    t = 1
    end = (config['end_time'] - config['start_time']) * steps
    while (t < end):
        arrival_prob = config['arrival_rates'][math.floor(t / steps)]
        request = generateRequest(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(request)
        serverNetwork.update(t)
        t += 1
    serverNetwork.listServers()
    serverNetwork.printRunningRequests(t, [0,1,2,3,4])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config',
                        dest='config',
                        help='Select config to run the simulation with',
                        default='default.yaml', type=str)

    args = parser.parse_args()

    with open(".\\simulation\\" + args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            run_simulation()
        except yaml.YAMLError as exc:
            print(exc)