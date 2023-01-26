import math
import random
from server_network.load_balancers.loadBalancer import LoadBalancer
import pandas as pd

def initServerState(id, config):
  ret = {}
  ret['id'] = id
  ret['capacity'] = config['max_processes']
  ret['active'] = []
  ret['num_running_requests'] = []
  ret['size_queue'] = []
  ret['num_finised_requests'] = []
  return ret

def initState(config):
    ret = {}
    ret['servers_capacity'] = config['max_processes']
    servers = {}
    for id in range(config['max_servers']):
      servers[id] = None
    ret['servers'] = servers
    ret['servers_used'] = []
    ret['profit'] = {
      'total': [],
      'rewards': [],
      'fails': [],
      'server_costs': [],   
    }
    ret['arrivals'] = {
      'small': [0],
      'large': [0],
    }
    return ret

class ServerNetwork:
  def __init__(self, num_servers, server_capacity, config, routing_policy = 'round_robbin', load_balancer = 'NN'):
    self.server_pointer = 0
    self.routing_policy = routing_policy
    self.capacity_servers = server_capacity
    self.all_servers = [Server(i, server_capacity, config) for i in range(config['max_servers'])]
    for server in self.all_servers:
      if server.id < num_servers:
        server.set_active() 
    self.active_servers = [server for server in self.all_servers if server.active]
    self.used_servers = []
    self.load_balancer = False if load_balancer == 'none' else LoadBalancer(load_balancer, config)
    self.prev_workload = 0
    self.config = config
    self.state_history = initState(config)

  def setConfig(self, config):
    self.config = config
    self.max_servers = config['max_servers']

  def getServer(self, id):
    for server in self.all_servers:
      if server.id == id:
        return server
    return False

  def removeServer(self):
    if len(self.active_servers) > 1:
      self.getServer(len(self.active_servers)-1).set_inactive()
      del self.active_servers[len(self.active_servers)-1]
    else:
      print('cannot have less than 1 server active')

  def addServer(self):
    if len(self.all_servers) > len(self.active_servers):
      server = self.all_servers[len(self.active_servers)]
      server.set_active()
      self.active_servers.append(server)
    else:
      print('Cannot have more than', self.max_servers, 'active')

  def setNActiveServers(self, n):
    diff = n - len(self.active_servers)
    if diff == 0:
      return
    for i in range(0, abs(diff)):
      self.addServer() if diff > 0 else self.removeServer()

  def getNextServer(self):
    if self.server_pointer >= len(self.active_servers):
      self.server_pointer = 0
    id = self.server_pointer
    self.server_pointer = (id + 1) if id < (len(self.active_servers) - 1) else 0
    return self.getServer(id)
  
  def updateState(self):
    for server in range(self.config['max_servers']):
      self.state_history['servers'][server] = self.getServer(server).updateState()
    self.state_history['servers_used'].append(len(self.active_servers))
    rewards, cost_servers, cost_failed, profit = self.calculate_profit_period()
    self.state_history['profit']['total'].append(profit)
    self.state_history['profit']['rewards'].append(rewards)
    self.state_history['profit']['fails'].append(cost_failed)
    self.state_history['profit']['server_costs'].append(cost_servers)
    self.state_history['arrivals']['small'].append(0)
    self.state_history['arrivals']['large'].append(0)

  def outputStateHistory(self):
    return self.state_history

  def evaluate(self, X_t = [], period = 0):
    self.used_servers.append(len(self.active_servers))
    self.updateState()
    self.reset_period()
    if self.load_balancer:
      num_servers = self.load_balancer.evaluate(X_t, period)
      self.setNActiveServers(num_servers)

  def train_lb(self, num_servers, X_t, profit):
    self.load_balancer.train(num_servers, X_t, profit)
  
  def get_profit_period(self, t = 0):
    self.reset_period()
    rewards, cost_servers, cost_failed, profit = self.calculate_profit_period()
    return len(self.active_servers), profit
  
  def reset_period(self):
    for server in self.all_servers:
      server.reset_period()

  def getTotalWorkload(self, t):
    workload = 0
    for server in self.all_servers:
      workload += server.getTotalWorkload(t)
    return workload

  def calculate_profit_period(self):
    cost_servers = -len(self.active_servers) * self.config['cost_server']
    rewards = 0
    cost_failed = 0
    for server in self.all_servers:
      for request in server.finished_requests_period:
        if request.completed:
          rewards += self.config['reward_small'] if request.type == 'small' else self.config['reward_large']
        elif request.failed:
          cost_failed -= self.config['cost_fail']
    # print('Cost servers:', cost_servers)
    # print('Rewards:', rewards)
    # print('Cost fails:', cost_failed)
    profit = cost_servers + rewards + cost_failed
    return rewards, cost_servers, cost_failed, profit
    

  def update(self, t):
    for server in self.all_servers:
      server.updateServer(t)
      
  def handleRequest(self, t, request):
    server = None
    self.state_history['arrivals'][request.type][-1] += 1
    if len(self.active_servers) == 0:
      self.addServer()
    if self.routing_policy == 'round_robin':
      server = self.getNextServer()
    elif self.routing_policy == 'least_connections':
      server = self.getLeastConnections()
    server.addRequest(t, request)

  def getLeastConnections(self):
    lc_server = self.active_servers[0]
    for server in self.active_servers:
      if len(lc_server.requests_running) > len(server.requests_running):
        lc_server = server
      elif len(lc_server.requests_running) == len(server.requests_running) and len(lc_server.queue) > len(server.queue):
        lc_server = server
    return lc_server
    
  def calculate_profit(self):
    cost_servers = -sum(self.used_servers) * self.config['cost_server']
    rewards = 0
    fails = 0
    for server in self.all_servers:
      for request in server.finished_requests:
        if request.completed:
          rewards += self.config['reward_small'] if request.type == 'small' else self.config['reward_large']
        elif request.failed:
          fails -= self.config['cost_fail']

    # print('Cost servers:', cost_servers)
    # print('Rewards:', rewards)
    # print('Cost fails:', fails)
    profit = cost_servers + rewards + fails
    return profit
          
  def listServers(self):
    print('Current status of all servers:\n')
    for server in self.all_servers:
      server.printStatus()

  def printRunningRequests(self, t, server_ids = [0]):
    print('Current running processes of all servers:\n')
    for id in server_ids:
      server = self.getServer(id)
      if (server):
        server.printRunningRequests(t)

class Server:
  def __init__(self, id, capacity, config):
    self.id = id
    self.capacity = capacity
    self.active = False
    self.requests_running = []
    self.queue = []
    self.finished_requests = []
    self.finished_requests_period = []
    self.state_history = initServerState(id, config)

  def updateState(self):
    self.state_history['active'].append(self.active)
    self.state_history['num_running_requests'].append(len(self.requests_running))
    self.state_history['size_queue'].append(len(self.queue))
    self.state_history['num_finised_requests'].append(len(self.finished_requests))
    return self.state_history

  def addRequest(self, t, request):
    request.setStartQueue(t)
    self.queue.append(request)
  
  def set_inactive(self):
    self.active = False

  def set_active(self):
    self.active = True

  def reset_period(self):
    self.finished_requests_period = []
    self.requests_running = []
    self.queue = []

  def getTotalWorkload(self, t):
    workload = 0
    for request in self.requests_running:
      workload += request.end - t
    for request in self.queue:
      workload += request.size
    return workload

  def updateServer(self, t):
    self.check_running_requests(t)
    self.check_queue(t)
    if self.capacity > len(self.requests_running):
      for i in range(0, self.capacity - len(self.requests_running)):
        if len(self.queue) > 0:
          request = self.queue[0]
          del self.queue[0]
          request.setEnd(t + request.size)
          request.setStart(t)
          self.requests_running.append(request)

  def check_running_requests(self, t):
    for i in reversed(range(len(self.requests_running))):
      request = self.requests_running[i]
      if t >= request.end:
        request.setCompleted(True)
        self.finished_requests.append(request)
        self.finished_requests_period.append(request)
        del self.requests_running[i]

  def check_queue(self, t):
    for i in reversed(range(len(self.queue))):
      request = self.queue[i]
      if t >= request.start_queue + request.max_age:
        request.setCompleted(False)
        request.setFailed(True)
        self.finished_requests.append(request)
        self.finished_requests_period.append(request)
        del self.queue[i]

  def printRunningRequests(self, t):
    if len(self.requests_running) > 0:
      result = 'Running requests of server: ' + str(self.id) + '\n'
      for request in self.requests_running:
        result += '\t type: ' + request.type
        result += ' - size: ' + str(request.size)
        result += ' - starting time: ' + str(request.start)
        result += ' - remaining running time: ' + str(request.end - t) + 's\n'
      print(result)

  def printStatus(self):
    result = 'Server ID: ' + str(self.id) + '\n'
    result += '\t Status: ' + ('active' if self.active else 'inactive') + '\n'
    result += '\t Remaining capacity: ' + str(self.capacity - len(self.requests_running)) + '\n'
    result += '\t Number of running processes: ' + str(len(self.requests_running)) + '\n'
    result += '\t Number of processes in queue: ' + str(len(self.queue)) + '\n'
    print(result)

# For testing (remove in final product):

# if __name__ == '__main__':
#   serverNetwork = ServerNetwork(3, 5)
#   serverNetwork.setNActiveServers(8)
#   serverNetwork.listServers()
#   serverNetwork.setNActiveServers(2)
#   serverNetwork.listServers()