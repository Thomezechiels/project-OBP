import sys
import random
from argparse import ArgumentParser
from itertools import cycle
import numpy as np

server_list = range(0, 10)
cur_server_load = np.random.randint(1,10,10)
server_cycle = cycle(server_list)
def round_robin(iter):
    return next(iter)

def random_load(servers):
    return random.choice(servers)

def LoadBalancer(type):
    if type == 'round_robin':
        return round_robin(server_cycle)
    elif type == 'random':
        return random_load(server_list)
    elif type == 'least_connections':
        return least_connections(server_list)

def least_connections(servers):
    lowest_value, lowest_server = min((val, idx) for (idx, val) in enumerate(cur_server_load))
    cur_server_load[lowest_server] += 1
    return servers[lowest_server]


def main():
    parser = ArgumentParser()

    parser.add_argument('-t', '--type',
                        dest='TYPE',
                        default='round_robin',
                        type=str)

    args = parser.parse_args()

    picked_servers = []
    for i in range(1,30):
        server_picked = LoadBalancer(args.TYPE)
        picked_servers.append(server_picked)
        print('Server picked: ' + str(server_picked))
    print("All servers: " + str(picked_servers))

if __name__ == '__main__':
    main()
    