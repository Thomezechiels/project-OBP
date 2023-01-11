import random

class ServerNetwork:
  def __init__(self, num_servers, server_capacity):
    self.num_servers = int(num_servers)
    self.server_pointer = 0
    self.capacity_servers = server_capacity
    servers = []
    for i in range(0, self.num_servers):
      servers.append(Server(i, server_capacity))
    self.servers = servers
    self.inactive_servers = []
    self.used_servers = []

  def update(self, t):
    for server in self.servers:
      server.updateServer(t)

  def getServer(self, id):
    for server in self.servers:
      if server.id == id:
        return server
    return False

  def removeServer(self):
    self.num_servers -= 1
    server = self.getServer(self.num_servers)
    server.set_inactive()
    self.inactive_servers.append(server)
    del self.servers[self.num_servers]

  def addServer(self):
    server = None
    
    if len(self.inactive_servers) > 0:
      server = self.inactive_servers[-1]
      server.set_active()
      del self.inactive_servers[-1]
    else:
      server = Server(self.num_servers, self.capacity_servers, )
    self.num_servers += 1
    self.servers.append(server)
    

  def getNextServer(self):
    id = self.server_pointer
    self.server_pointer = (id + 1) if id < (self.num_servers - 1) else 0
    return self.getServer(id)

  def evaluate(self):
    #implement the algorithm here. It can add or remove servers just before the next period starts
    if random.random() < 0.4: #this is just here temporarily for testing.
      self.addServer() 

    self.used_servers.append(self.num_servers)

  def handleRequest(self, t, request):
    server = self.getNextServer()
    server.addRequest(t, request)

  def calculate_profit(self, reward_small, reward_large, cost_fail, cost_server):
    profit = -sum(self.used_servers) * cost_server
    for server in self.servers:
      for request in server.finished_requests:
        if request.completed:
            profit += reward_small if request.type == 'small' else reward_large
        else:
          print(cost_fail)
          profit -= cost_fail
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

  def addRequest(self, t, request):
    request.setStartQueue(t)
    self.queue.append(request)
  
  def set_inactive(self):
    self.active = False

  def set_active(self):
    self.active = True

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
        del self.requests_running[i]
      elif t >= request.start + request.max_age:
        request.setCompleted(False)
        self.finished_requests.append(request)
        del self.requests_running[i]

  def check_queue(self, t):
    for i in reversed(range(len(self.queue))):
      request = self.queue[i]
      if t >= request.start_queue + request.max_age:
        request.setCompleted(False)
        self.finished_requests.append(request)
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
#   serverNetwork.listServers()
#   serverNetwork.removeServer()
#   serverNetwork.removeServer()
#   serverNetwork.addServer()
#   serverNetwork.inactive_servers[0].printStatus()
#   serverNetwork.listServers()