import math
import random
from server_network.load_balancers.loadBalancer import LoadBalancer

random.seed(10)

class ServerNetwork:
  def __init__(self, num_servers, server_capacity, routing_policy = 'round_robbin', load_balancer = 'none'):
    self.num_servers = int(num_servers)
    self.server_pointer = 0
    self.routing_policy = routing_policy
    self.capacity_servers = server_capacity
    servers = [Server(i, server_capacity) for i in range(num_servers)]
    self.servers = servers
    self.inactive_servers = []
    self.used_servers = []
    self.load_balancer = LoadBalancer(load_balancer)
    self.prev_workload = 0

  def setConfig(self, config):
    self.config = config
    self.max_servers = config['max_servers']

  def getServer(self, id):
    for server in self.servers:
      if server.id == id:
        return server
    return False

  def removeServer(self):
    if self.num_servers > 1:
      self.num_servers -= 1
      server = self.getServer(self.num_servers)
      server.set_inactive()
      self.inactive_servers.append(server)
      del self.servers[self.num_servers]
    else:
      print('cannot have less than 1 server active')

  def addServer(self):
    server = None
    if self.num_servers < self.max_servers:
      if len(self.inactive_servers) > 0:
        server = self.inactive_servers[-1]
        server.set_active()
        del self.inactive_servers[-1]
      else:
        server = Server(self.num_servers, self.capacity_servers, )
      self.num_servers += 1
      self.servers.append(server)
    else:
      print('Cannot have more than', self.max_servers, 'active')

  def setNActiveServers(self, n):
    diff = n - len(self.servers)
    if diff == 0:
      return
    else:
      for i in range(0, abs(diff)):
        self.addServer() if diff > 0 else self.removeServer()

  def getNextServer(self):
    if self.server_pointer >= self.num_servers:
      self.server_pointer = 0
    id = self.server_pointer
    self.server_pointer = (id + 1) if id < (self.num_servers - 1) else 0
    return self.getServer(id)

  def evaluate(self, t, arrivals):
    self.used_servers.append(self.num_servers)
    if self.load_balancer.model:
      num_servers = self.load_balancer.evaluate({'arrivals': arrivals, 'workload': int(self.getTotalWorkload(t))})
      self.setNActiveServers(num_servers)
  
  def data_generation_evalutation(self, t = 0):
    workload = self.prev_workload
    self.prev_workload = self.getTotalWorkload(t)
    profit_period = self.calculate_profit_period()
    self.reset_period()
    num_servers_used = self.num_servers
    num_servers = random.randint(1, self.config['max_servers'])
    self.setNActiveServers(num_servers)
    return num_servers_used, profit_period, workload
  
  def reset_period(self):
    for server in self.servers:
      server.reset_period()

  def getTotalWorkload(self, t):
    workload = 0
    for server in self.servers:
      workload += server.getTotalWorkload(t)
    return workload

  def calculate_profit_period(self):
    cost_servers = -self.num_servers * self.config['cost_server']
    rewards = 0
    cost_failed = 0
    for server in self.servers:
      for request in server.finished_requests_period:
        if request.completed:
          rewards += self.config['reward_small'] if request.type == 'small' else self.config['reward_large']
        elif request.failed:
          cost_failed -= self.config['cost_fail']
    # print('Cost servers:', cost_servers)
    # print('Rewards:', rewards)
    # print('Cost fails:', cost_failed)
    profit = cost_servers + rewards + cost_failed
    return profit
    

  def update(self, t):
    for server in self.servers:
      server.updateServer(t)
      
  def handleRequest(self, t, request):
    server = None
    if self.num_servers == 0:
      self.addServer()
    if self.routing_policy == 'round_robin':
      server = self.getNextServer()
    elif self.routing_policy == 'least_connections':
      server = self.getLeastConnections()
    server.addRequest(t, request)

  def getLeastConnections(self):
    lc_server = self.servers[0]
    for server in self.servers:
      if len(lc_server.requests_running) > len(server.requests_running):
        lc_server = server
      elif len(lc_server.requests_running) == len(server.requests_running) and len(lc_server.queue) > len(server.queue):
        lc_server = server
    return lc_server
    
  def calculate_profit(self):
    cost_servers = -sum(self.used_servers) * self.config['cost_server']
    rewards = 0
    fails = 0
    servers = self.servers + self.inactive_servers
    for server in servers:
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
    for server in self.servers:
      server.printStatus()

  def printRunningRequests(self, t, server_ids = [0]):
    print('Current running processes of all servers:\n')
    for id in server_ids:
      server = self.getServer(id)
      if (server):
        server.printRunningRequests(t)

class Server:
  def __init__(self, id, capacity):
    self.id = id
    self.capacity = capacity
    self.active = True
    self.requests_running = []
    self.queue = []
    self.finished_requests = []
    self.finished_requests_period = []

  def addRequest(self, t, request):
    request.setStartQueue(t)
    self.queue.append(request)
  
  def set_inactive(self):
    self.active = False

  def set_active(self):
    self.active = True

  def reset_period(self):
    self.finished_requests_period = []

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