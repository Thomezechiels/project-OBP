class ServerNetwork:
  def __init__(self, num_servers, server_capacity):
    self.num_servers = int(num_servers)
    self.server_pointer = 0
    servers = []
    for i in range(0, self.num_servers):
      servers.append(Server(i, server_capacity))
    self.servers = servers

  def update(self, t):
    for server in self.servers:
      server.updateServer(t)

  def getServer(self, id):
    for server in self.servers:
      if server.id == id:
        return server
    return False

  def listServers(self):
    print('Current status of all servers:\n')
    for server in self.servers:
      server.printStatus()

  def getNextServer(self):
    id = self.server_pointer
    self.server_pointer = (id + 1) if id < (self.num_servers - 1) else 0
    return self.getServer(id)

  def handleRequest(self, request):
    server = self.getNextServer()
    server.addRequest(request)

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
    self.requests_running = []
    self.queue = []
    self.finished_requests = []

  def addRequest(self, request):
    self.queue.append(request)

  def updateServer(self, t):
    self.check_running_requests(t)
    if self.capacity > len(self.requests_running):
      for i in range(0, self.capacity - len(self.requests_running)):
        if len(self.queue) > 0:
          # print('add request')
          request = self.queue[0]
          del self.queue[0]
          request.setEnd(t + request.size)
          request.setStart(t)
          self.requests_running.append(request)

  def check_running_requests(self, t):
    for i in reversed(range(len(self.requests_running))):
      request = self.requests_running[i]
      if t >= request.end:
        self.finished_requests.append(request)
        del self.requests_running[i]

  def printRunningRequests(self, t):
    result = 'Running requests of server: ' + str(self.id) + '\n'
    for request in self.requests_running:
      result += '\t type: ' + request.type
      result += ' - size: ' + str(request.size)
      result += ' - starting time: ' + str(request.start)
      result += ' - remaining running time: ' + str(request.end - t) + 's\n'
    print(result)

  def printStatus(self):
    result = 'Server ID: ' + str(self.id) + '\n'
    result += '\t Remaining capacity: ' + str(self.capacity - len(self.requests_running)) + '\n'
    result += '\t Number of running processes: ' + str(len(self.requests_running)) + '\n'
    result += '\t Number of processes in queue: ' + str(len(self.queue)) + '\n'
    print(result)

if __name__ == '__main__':
  serverNetwork = ServerNetwork(5, 5)
  for i in range(0, 10):
    serverNetwork.getNextServer()