import numpy as np

class TabularQ:
    
    def __init__(self, shape, init_value = 0.6):
        self.data = np.full(shape, init_value, dtype=np.float64)
        self.n_states = np.prod(shape[:-1])
        self.n_actions = shape[-1]    
        
    def of(self, state, action):
        return self.data[state][action]
    
    def all_values(self, state):
        return self.data[state]
    
    def update(self, state, action, delta):
        self.data[state][action] += delta
    
    