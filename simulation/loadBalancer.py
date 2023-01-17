class LoadBalancer:
    def __init__(self):
        self.network_state = None

    #load balancer takes the current state of the network and outputs the number of servers to use next period
    def evaluate(model,config, self):
        return model.determineservers(config, 0.2)
