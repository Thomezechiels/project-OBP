from argparse import ArgumentParser
from pathlib import Path 
import yaml
import pickle
import json
from statistics import mean

import math
import random
import numpy as np
import pandas as pd

from server_network.request import Request
from server_network.servers import ServerNetwork

global config

def generateRequest(arrival_prob):
    if random.random() <= arrival_prob:
        type = 'small' if random.random() < config['prob_small'] else 'large'
        size = np.random.normal(loc=config['mean_' + type], scale=config['std_' + type])
        max_age = config['max_wait_' + type]
        return Request(type, size, max_age)
    else:
        return False    
    
def run_simulation():
    serverNetwork = ServerNetwork(5, config['max_processes'], config = config, routing_policy='round_robin', load_balancer='none')
    serverNetwork.setConfig(config)
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    while (t < end):
        arrival_prob = config['arrival_rates'][math.floor(t / steps)]
        if (t / steps).is_integer():
            serverNetwork.evaluate(t, arrival_prob)
        request = generateRequest(arrival_prob)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        t += 1
    
    history = serverNetwork.outputStateHistory()
    print(len(history['servers'][0]['num_finised_requests']))
    # for server in history['servers']:
    #     print(len(server['num_finised_requests']), len(server['size_queue']), len(server['num_running_requests']) )
    # with open("network_history.pickle", "wb") as handle:
    #     pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_arrivals():
    requests_list = []
    t = 0
    steps = config['steps']
    end = (config['end_time'] - config['start_time']) * steps
    hour = 0
    while (t < end):
        arrival_prob = config['arrival_rates'][math.floor(t / steps)]
        request = generateRequest(arrival_prob)
        if (t / steps).is_integer():
            hour +=1
        if request == False:
            requests_list.append([0,hour])
        else:
            requests_list.append([1,hour])
        t += 1

    requests_df = pd.DataFrame(requests_list, columns = ['request','hour'])
    requests_df["index"] = range(1, len(requests_df) + 1)
    requests_df = requests_df.set_index("index")

    filepath = Path('data/requests_df.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    requests_df.to_csv(filepath)

def estimate_arrival_rate(hour):
    df = pd.read_csv('data/requests_df.csv')
    count = sum(df.loc[df['hour'] == hour]['request'])
    rate = count/config['steps']
    return rate

def calculate_arrival_rates():
    end = config['end_time']
    start = config['start_time']
    arrival_rates_est = []
    for i in range(start+1,end+1):
        estimated_arrival_rate = estimate_arrival_rate(i)
        arrival_rates_est.append([i,estimated_arrival_rate])
    arrival_rates_df = pd.DataFrame(arrival_rates_est, columns = ['hour','rate'])
    arrival_rates_df["index"] = range(1, len(arrival_rates_df) + 1)
    arrival_rates_df = arrival_rates_df.set_index("index")
    return arrival_rates_df


def calculate_p_hats(runs):
    rate_dict = {}
    arrival_rates_df = calculate_arrival_rates()
    for i in range(len(arrival_rates_df['hour'])):
        rate_dict[i+1] = []
    for i in range(runs):
        save_arrivals()
        arrival_rates_df = calculate_arrival_rates()
        for index,row in arrival_rates_df.iterrows():
            rate_dict[index].append(row['rate'])    
    for index,row in arrival_rates_df.iterrows():
        arrival_rates_df.loc[index,'rate'] = mean(rate_dict[index])
        
    obj = pd.read_pickle(r'data/server_optimum.pickle')
    for index, row in arrival_rates_df.iterrows():
        rounded_rate = np.round(row['rate'],2)
        arrival_rates_df.loc[index,'servers'] = obj[rounded_rate]
    return arrival_rates_df

def run_data_simulation():
    serverNetwork = ServerNetwork(6, config['max_processes'], routing_policy='round_robin')
    serverNetwork.setConfig(config)
    data_points = []
    steps = config['steps']
    t = 0
    end = (config['end_time'] - config['start_time']) * steps
    while (t < end + 1):
        if (t / steps).is_integer():
            if t > 0:
                arrival = config['arrival_rates'][math.floor((t-1)/ steps)]
                num_servers, profit, workload = serverNetwork.data_generation_evalutation(t = t)
                data_point = {
                    "action": num_servers,
                    "cost": -profit,
                    "probability": 0.1,
                    "feature_arrival": arrival,
                    "feature_workload": int(round(workload, -3)),
                }
                data_points.append(data_point)
            elif t == 0:
                num_servers, profit, workload = serverNetwork.data_generation_evalutation(t = t)
        if t < end:
            arrival_prob = config['arrival_rates'][math.floor(t / steps)]
            request = generateRequest(arrival_prob, config)
            if (request and request.size > 0):
                serverNetwork.handleRequest(t, request)
            serverNetwork.update(t)
        t += 1
    return data_points

def generateData():
    train_data = []
    for i in range(10):
        data_points = run_data_simulation()
        train_data += data_points
        print('Finished run:', i)
    
    train_df = pd.DataFrame(train_data)
    train_df["index"] = range(1, len(train_df) + 1)
    train_df = train_df.set_index("index")

    filepath = Path('data/training_test.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    train_df.to_csv(filepath)

def run_simulation_hour(n, arrival_rate):
    serverNetwork = ServerNetwork(n, config['max_processes'], routing_policy='round_robin', load_balancer='none')
    serverNetwork.setConfig(config)
    steps = config['steps']
    end = steps * 1
    t = 0
    while (t < end):
        request = generateRequest(arrival_rate)
        if (request and request.size > 0):
            serverNetwork.handleRequest(t, request)
        serverNetwork.update(t)
        if (t / steps - 1).is_integer():
            serverNetwork.evaluate(t, arrival_rate)
        t += 1
    return serverNetwork.calculate_profit()

def calculate_profit_n(n, arrival_rate):
    profit = 0
    RUNS = 10
    for j in range(RUNS):
        profit += run_simulation_hour(n, arrival_rate)
    return profit/RUNS

def binary_search(low, high, arrival_prob):
    cache = {}
    def _reward_func(x, arrival_prob):
        if x <= 0:
            return 0
        elif x not in cache:
            cache[x] = calculate_profit_n(x, arrival_prob)
        return cache[x]
        
    while (high - low) > 0:
        if high - low == 1:
            return high if _reward_func(high, arrival_prob) > _reward_func(low, arrival_prob) else low
        mid = int((low + high) / 2)
        neighborhood = [mid-1, mid, mid+1]
        rewards = [_reward_func(x, arrival_prob) for x in neighborhood]
        max_index = rewards.index(max(rewards))
        max_value = neighborhood[max_index]

        if max_value == mid:
            return max_value
        elif max_value < mid:
            high = mid
        else:
            low = mid

    return mid

def arrival_server_function():
    d = {}
    for i in range(1, 101):
        arrival_prob = i / 100
        print('Finding optimum for arrival rate:', arrival_prob)
        d[arrival_prob] = binary_search(1, 11, arrival_prob)
    with open("server_optimum.pickle", "wb") as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config',
                        dest='config',
                        help='Select config to run the simulation with',
                        default='default.yaml', type=str)

    args = parser.parse_args()
    filepath = Path(args.config)  
    filepath.parent.mkdir(parents=True, exist_ok=True)  

    # with open("server_optimum.pickle", "rb") as handle:
    #     my_dict = pickle.load(handle)
    #     print(json.dumps(my_dict, indent=4))

    with open(filepath, "r") as stream:
        try:
            random.seed(10)
            config = yaml.safe_load(stream)
            # calculate_p_hats(5)
            run_simulation()
            # generateData()
            # arrival_server_function()
        except yaml.YAMLError as exc:
            print(exc)