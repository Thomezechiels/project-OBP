class Request:
    def __init__(self, type, size):
        self.type = type
        self.size = size

    def setEnd(self, time):
        self.end = time

    def setStart(self, time):
        self.start = time