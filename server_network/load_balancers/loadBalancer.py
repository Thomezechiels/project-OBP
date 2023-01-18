from server_network.load_balancers.ContextualBandit import ContextualBandit

class LoadBalancer:
  def __init__(self, load_balancer):
        if (load_balancer == 'contextual_bandit'):
          self.model = ContextualBandit('--cb_explore 10 --cover 8')
        elif (load_balancer == 'regression'):
          self.model = None

  def evaluate(self, state):
    return self.model.evaluate(state)