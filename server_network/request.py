class Request:
    def __init__(self, type, size, max_age):
        self.type = type
        self.size = size
        self.max_age = max_age
        self.failed = False

    def setEnd(self, time):
        self.end = time

    def setFailed(self, failed):
        self.failed = failed

    def setStartQueue(self, time):
        self.start_queue = time
        
    def setStart(self, time):
        self.start = time
    
    def setCompleted(self, completed):
        self.completed = completed