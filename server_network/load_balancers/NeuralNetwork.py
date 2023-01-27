import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import random

class NeuralNetwork:
    def __init__(self):
        self.model = False
        self.data = False
        self.iteration = 0

    def evaluate(self, X_t):        
        # max_profit = 0
        # optimal_servers = 1
        # for num_server in range(1,10):
        #     X = [num_server] + list(X_t)
        #     profit = self.model.predict([X])
        #     # print(num_server, X, profit)
        #     if profit[0] > max_profit:
        #         max_profit = profit[0]
        #         optimal_servers = num_server
        # print('max_profit', max_profit)
        # print('optimal_servers', optimal_servers)
        return random.randint(1,10)
        return optimal_servers
        
    
    def train(self, num_servers, X_t, profit):
        return
        # X = [num_servers] + list(X_t) 
        # if not self.data:
        #     self.data = {
        #         'X': [X],
        #         'profit': [profit]
        #     }
        # else:
        #     self.data['X'].append(X)
        #     self.data['profit'].append(profit)

        # if not self.model:
        #     self.model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', random_state=10, learning_rate_init=0.01, max_iter=2000)
        # self.model.fit(self.data['X'], self.data['profit'])
        # print([X], [profit])