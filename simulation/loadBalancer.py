class LoadBalancer:
    def __init__(self):
        self.network_state = None

    #load balancer takes the current state of the network and outputs the number of servers to use next period
    def evaluate(self, state):
        self.state = state
        num_servers = run_algorithm(state) #implement the alogrithm here
        return num_servers

        