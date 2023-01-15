

def generateRequest(config, arrival_prob):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        return Request(type, size, config['max_wait'])
    else:
        return False    

def run_simulation(config):
    serverNetwork = ServerNetwork(1, config['max_processes'])
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    while (t < end):
        arrival_prob = config['arrival_rates'][math.floor(t / steps)]
        if (t / steps).is_integer():
            serverNetwork.evaluate()
        request = generateRequest(config, arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1

    return serverNetwork.outputState()
    # serverNetwork.listServers()
    # serverNetwork.printRunningRequests(t, [0,1,2,3,4])
    # print('profit:', serverNetwork.calculate_profit(config['reward_small'], config['reward_large'], config['cost_fail'], config['cost_server']))

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('-C', '--config',
#                         dest='config',
#                         help='Select config to run the simulation with',
#                         default='default.yaml', type=str)

#     args = parser.parse_args()

#     SCRIPT_PATH = str(pathlib.Path(__file__).parent.parent.resolve())
#     path = os.path.join(SCRIPT_PATH, os.path.join(args.config))
    
#     with open(path, "r") as stream:
#         try:
#             config = yaml.safe_load(stream)
#             run_simulation()
#         except yaml.YAMLError as exc:
#             print(exc)