import vowpalwabbit
import numpy as np  
import random

class LoadBalancer:
  def __init__(self):
        self.network_state = None

    # def sample_custom_pmf(self, pmf):
    #     total = sum(pmf)
    #     scale = 1 / total
    #     pmf = [x * scale for x in pmf]
    #     draw = random.random()
    #     sum_prob = 0.0
    #     for index, prob in enumerate(pmf):
    #         sum_prob += prob
    #         if(sum_prob > draw):
    #             return index + 1    
  def evaluate(self, state):
    vw = vowpalwabbit.Workspace("--cb_explore 10 --cover 8 -i cb.model", quiet=True)
    test_example = "| " + str(state['arrivals']) + " " + str(round(state['workload'], -3))
    choice = vw.predict(test_example)
    # choice = self.sample_custom_pmf(choice)
    choice = np.argmax(choice) + 1
    return choice