class ReplayBuffer:
    def __init__(self):
        self.buffer = {}
        self.index = 0

    def store(self, observation):
        self.buffer[self.index] = {}
        for k in observation.keys():
            self.buffer[self.index] = {k : observation[k]}
        self.buffer +=1
    
    def sample(self, index, metric):
        metrics = self.buffer.get(index,None)
        if metrics is None:
            raise KeyError(f'Index {index} is not present inside the buffer')
    
        value = metrics.get(metric,None)
        if value is None:
            raise KeyError(f'Metric {metric} is not present inside the buffer')
        return value
    
    def empty(self):
        self.buffer = {}
        self.index = 0