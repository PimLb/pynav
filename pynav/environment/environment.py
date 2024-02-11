import numpy as np

from typing import Tuple, Dict



class Environment:
    
    def __init__(self, 
                 data : np.ndarray, 
                 source_position : np.ndarray,
                 source_radius : float = 20, # source radius
                 bounds : np.ndarray = None,
                 rewards : Tuple[float, float] = (1.0, -0.01), # reward at source and in other positions
                 boundary_conditions : str = None,
                 seed : int = 1212121314,
                 ):
        self.data = data
        self.boundary_conditions = boundary_conditions
        self.rnd_state = np.random.RandomState(seed)
        self.bounds = bounds # [[min_x, max_x], [min_y, max_y]]
        if bounds is None:
            self.bounds = np.asarray([
                [0, self.data.shape[1]],
                [0, self.data.shape[2]],
            ])
        self.source_position = source_position
        self.source_radius = source_radius
        self.rewards = rewards
        self.MAX_T = self.data.shape[0] 
                
    def _is_in_bounds(self, pos):
        return self.bounds[0, 0] <= pos[0] < self.bounds[0, 1] and self.bounds[1, 0] <= pos[1] < self.bounds[1, 1]

    def _is_in_domain(self, pos):
        return 0 <= pos[0] < self.data.shape[1] and 0 <= pos[1] < self.data.shape[2]

    def source_reached(self, pos):
        return np.sum((pos - self.source_position) ** 2) <= (self.source_radius ** 2)
        
    def get_observation(self, pos, t): 
        # if the agent is inside the data grid, 
        # returns the data point in its current position
        # otherwise, it returns 0
        if self._is_in_domain(pos):
            return self.data[t : t + 1, pos[0], pos[1]]
        return np.zeros(1)

                    
    def make_move(self, pos, action : np.ndarray):
        pos = pos + action
        if self.boundary_conditions in ['wrap', 'wrap_vertical']:    
            if pos[0] < self.bounds[0, 0]:
                pos[0] = self.bounds[0, 1] - 1
            elif pos[0] >= self.bounds[0, 1]:
                pos[0] = self.bounds[0, 0]
        if self.boundary_conditions in ['wrap', 'wrap_horizontal']:    
            if pos[1] < self.bounds[1, 0]:
                pos[1] = self.bounds[1, 1] - 1
            elif pos[1] >= self.bounds[1, 1]:
                pos[1] = self.bounds[1, 0]        
        terminated = self.source_reached(pos)
        reward = self.rewards[0] if terminated else self.rewards[1]
        return pos, reward, terminated

        
        