from server_network.load_balancers.ContextualBandit import ContextualBandit
from server_network.load_balancers.NeuralNetwork import NeuralNetwork
import math
import random

class LoadBalancer:
  def __init__(self, load_balancer, config):
        self.config = config
        if (load_balancer == 'none'):
           self.model = False
        elif load_balancer == 'NN':
           self.model == NeuralNetwork()
        elif (load_balancer == 'contextual_bandit'):
          self.model = ContextualBandit('--cb_explore 10 --cover 8')
        elif (load_balancer == 'regression'):
          self.model = None

  def evaluate(self, X_t, t):
    epsilon = 1 / math.sqrt(t)
    if random.random() < epsilon:
       return random.randint(0, self.config['max_servers'])
    return self.model.evaluate(X_t)
    
  def train(self, num_servers, X_t, profit):
     self.model.train(num_servers, X_t, profit)

    