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
import scipy.stats

from server_network.request import Request
from server_network.servers import ServerNetwork

import joblib

global config

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

def generateRequest(arrival_prob):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False    
    
def run_simulation():
    serverNetwork = ServerNetwork(5, config['max_processes'], config = config, routing_policy='round_robin', load_balancer='rf')
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    arrival_prob = 0

    context_vector_options_max = np.matrix([[1, 1, 1, 1/5]])
    context_vector_options_min = np.matrix([[1/7, scipy.stats.norm(0.5, 0.2).pdf(1/24)/2, 1/4, 1]])
    weights = np.matrix([[4],[8],[3],[-2]])
    max = multiply_matrix(context_vector_options_max, weights)[0][0]
    min = multiply_matrix(context_vector_options_min, weights)[0][0]
    
    while (t < end):
        period = t / steps
        if (period).is_integer():
            if t > 0:
                num_servers, profit = serverNetwork.get_profit_period(t = t)
                serverNetwork.train_lb(num_servers, X_t, profit)
            # X_t = np.random.normal(0, 0.6, 8)
            # arrival_prob = 1 / (1 + math.exp(-X_t.sum())

            hour_context = scipy.stats.norm(0.5, 0.2).pdf(((period + 1) % 24)/24)/2
            X_t = np.array([random.randint(1, 7)/7, hour_context, random.randint(1, 4)/4, random.randint(1, 5)/5])
            arrival_prob = ((multiply_matrix(np.asmatrix(X_t), weights)[0][0] - min)/(max-min))*0.8 + 0.1

            serverNetwork.evaluate(X_t, period)
        request = generateRequest(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1
    
    filename = 'neural_network.sav'
    joblib.dump(serverNetwork.load_balancer.model.model, filename)

if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

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