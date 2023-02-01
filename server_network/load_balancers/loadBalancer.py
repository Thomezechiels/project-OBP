from server_network.load_balancers.SimpleRegression import SimpleRegression
from server_network.load_balancers.NeuralNetwork import NeuralNetwork
from server_network.load_balancers.RandomForest import RandomForest
import math
import random

class LoadBalancer:
   def __init__(self, load_balancer, config):
      self.config = config
      if load_balancer == 'Neural Network':
         self.model = NeuralNetwork()
      elif (load_balancer == 'Simple Regression'):
         self.model = SimpleRegression()
      elif (load_balancer == 'Random Forest'):
         self.model = RandomForest()
      else:
         self.model = False

   def evaluate(self, X_t, t):
      # t = 1 if t < 1 else t
      # epsilon = 1 / math.sqrt(t)
      # if random.random() < epsilon:
      #    return random.randint(1, self.config['max_servers'])
      return self.model.evaluate(X_t)
   
   def evaluate_live(self, X_t, t):
      return self.model.evaluate_live(X_t)
    
   def train(self, num_servers, X_t, profit):
      self.model.train(num_servers, X_t, profit)

    