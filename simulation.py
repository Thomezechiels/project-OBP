from pathlib import Path 
import yaml
import joblib
from argparse import ArgumentParser

import random
import numpy as np
import scipy.stats

from server_network.request import Request
from server_network.servers import ServerNetwork

global config

def multiply_matrix(A, B):
    """
    This function multiplies two matrices A and B and returns the result.
    The function raises an error if the number of columns in A is not equal to the number of rows in B.

    Args:
    A (numpy.ndarray): The first matrix
    B (numpy.ndarray): The second matrix

    Returns:
    numpy.ndarray: The product of the two matrices A and B
    """
    global C
    if A.shape[1] == B.shape[0]:
        rows = B.shape[1]
        cols = A.shape[0]
        C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
        for row in range(rows):
            for col in range(cols):
                for elt in range(len(B)):
                    C[row, col] += A[row, elt] * B[elt, col]
        return C
    else:
        return np.array([[20]])


def generate_request(arrival_prob):
    """
    This function generates a request with a probability given by the argument 'arrival_prob'.
    The type of request is either 'small' or 'large' with probabilities given by the global
    'config' dictionary. The size of the request is sampled from a normal distribution with mean
    and standard deviation dependent on its type. The maximum age of the request
    is also specified in the 'config' dictionary.

    Args:
    arrival_prob (float): The probability of generating a request

    Returns:
    Request or False: A Request object if a request is generated, False otherwise.
    """
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(
            loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False
    
def init_context_vector():
    """
    This function initializes the context vector by computing the context vector options minimum and maximum using matrix multiplication with a given set of weights. It then returns the weights, the minimum and the maximum.

    Returns:
    Tuple: Tuple containing three values: the weights, the minimum and the maximum.
    """
    context_vector_options_max = np.matrix([[1, 1, 1, 1/5]])
    context_vector_options_min = np.matrix(
        [[1/7, scipy.stats.norm(0.5, 0.2).pdf(1/24)/2, 1/4, 1]])
    weights = np.matrix([[4], [8], [3], [-2]])
    min = multiply_matrix(context_vector_options_min, weights)[0][0]
    max = multiply_matrix(context_vector_options_max, weights)[0][0]
    return weights, min, max

def generate_context_arrival(period, weights, min, max):
    """
    This function generates the context arrival by first computing the hour context based on a normal distribution and a given period. It then generates the other context values as random variables, computes the arrival probability using a formula that involves the context values and the weights, and returns the context and the arrival probability.

    Parameters:
    period (int): The period for which to generate the context arrival.
    weights (numpy.ndarray): The weights to be used in the computation of the arrival probability.
    min (numpy.ndarray): The minimum value used in the computation of the arrival probability.
    max (numpy.ndarray): The maximum value used in the computation of the arrival probability.

    Returns:
    Tuple: Tuple containing two values: the context as a numpy array, and the arrival probability as a float.
    """
    hour_context = scipy.stats.norm(0.5, 0.2).pdf(((period + 1) % 24)/24)/2
    X_t = np.array([random.randint(1, 7)/7, hour_context, random.randint(1, 4)/4, random.randint(1, 5)/5])
    arrival_prob = ((multiply_matrix(np.asmatrix(X_t), weights)[0][0] - min)/(max-min))*0.8 + 0.1
    return X_t, arrival_prob  
    
def run_simulation(model, routing, output):
    t = 0
    steps = config['steps']
    end = (config['end_time'] - config['start_time']) * steps
    weights, min, max = init_context_vector()
    
    serverNetwork = ServerNetwork(5, config['max_processes'], config=config, routing_policy=routing, load_balancer=model)
        
    while (t < end):
        period = t / steps
        if period.is_integer():
            if t > 0:
                num_servers, profit = serverNetwork.get_profit_period(t = t)
                serverNetwork.train_lb(num_servers, X_t, profit)
            X_t, arrival_prob = generate_context_arrival(period, weights, min, max)
            serverNetwork.evaluate(X_t, period)
        request = generate_request(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1

    #Save trained model:
    joblib.dump(serverNetwork.load_balancer.model.model, output)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config',
                        dest='config',
                        help='Select config to run the simulation with',
                        default='config.yaml', type=str)
    parser.add_argument('-M', '--model',
                        dest='model',
                        choices=['Decision Tree', 'Random Forest', 'Neural Network'],
                        help='Choose the model to train',
                        default='Decision Tree', type=str)
    parser.add_argument('-R', '--routing',
                        dest='routing',
                        choices=['Round Robin', 'Least Connections'],
                        help='Choose the routing policy to use',
                        default='Round Robin', type=str)
    parser.add_argument('-O', '--output',
                        dest='output',
                        help='The filename of the file in which the resulting model will be stored',
                        default='model.sav', type=str)

    args = parser.parse_args()
    filepath = Path(args.config)  
    filepath.parent.mkdir(parents=True, exist_ok=True)  

    with open(filepath, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            run_simulation(args.model, args.routing, args.output)
        except yaml.YAMLError as exc:
            print(exc)