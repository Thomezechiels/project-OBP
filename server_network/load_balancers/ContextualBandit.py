import vowpalwabbit
import numpy as np  
import random

class ContextualBandit:
    def __init__(self, model):
        self.vw = vowpalwabbit.Workspace(model, quiet=True)

    def sample_custom_pmf(self, pmf):
        total = sum(pmf)
        scale = 1 / total
        pmf = [x * scale for x in pmf]
        draw = random.random()
        sum_prob = 0.0
        for index, prob in enumerate(pmf):
            sum_prob += prob
            if(sum_prob > draw):
                return index + 1   

    def evaluate(self, state, use_pmf=False):
        vw = vowpalwabbit.Workspace("--cb_explore 10 --cover 8 -i cb.model", quiet=True)
        test_example = "| " + str(state['arrivals']) + " " + str(round(state['workload'], -3))
        choice = vw.predict(test_example)
        choice = np.argmax(choice) + 1
        print(test_example, choice)
        return choice