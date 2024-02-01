import numpy as np

from typing import Tuple, Dict

from pynav.utils import default_reward

COORDS = {
    0 : np.asarray([ 1, 0], dtype=np.int32), 
    1 : np.asarray([-1, 0], dtype=np.int32), 
    2 : np.asarray([0,  1], dtype=np.int32), 
    3 : np.asarray([0, -1], dtype=np.int32), 
}

INV_ACTIONS = [1, 0, 3, 2]

class Environment:
    
    def __init__(self, 
                 data : np.ndarray, 
                 source_position : np.ndarray,
                 source_radius : float = 20, # source radius
                 velocity : int = 10,
                 bounds : np.ndarray = None,
                 rewards : Tuple[float, float] = (1.0, -0.01), # reward at source and in other positions
                 seed : int = 1212121314,
                 ):
        self.data = data
        self.rnd_state = np.random.RandomState(seed)
        self.bounds = bounds # [[min_x, max_x], [min_y, max_y]]
        self.velocity = velocity
        if bounds is None:
            self.bounds = np.asarray([
                [0, self.data.shape[0]],
                [0, self.data.shape[1]],
            ])
        self.source_position = source_position
        self.source_radius = source_radius
        self.rewards = rewards
        self.t = 0
        self.iters = 0
        self.MAX_T = self.data.shape[0] 
                
    def _is_in_bounds(self, pos):
        return self.bounds[0, 0] <= pos[0] < self.bounds[0, 1] and self.bounds[1, 0] <= pos[1] < self.bounds[1, 1]

    def _is_in_domain(self, pos):
        return 0 <= pos[0] < self.data.shape[0] and 0 <= pos[1] < self.data.shape[1]

    def _is_near_enough(self, pos):
        return np.linalg.norm(pos - self.source_position) <= self.source_radius

    def source_reached(self):
        return np.linalg.norm(self.current_position - self.source_position) <= self.source_radius
        
    def clip_position(self, pos):
        new_pos = pos.copy()
        if pos[0] < self.bounds[0, 0]:
            delta_x = self.bounds[0, 0] - pos[0]
            new_pos[0] = self.bounds[0, 1] - delta_x
        elif pos[0] >= self.bounds[0, 1]:
            delta_x = pos[0] - self.bounds[0, 1]
            new_pos[0] = self.bounds[0, 0] + delta_x
        if pos[1] < self.bounds[1, 0]:
            delta_x = self.bounds[1, 0] - pos[1]
            new_pos[1] = self.bounds[1, 1] - delta_x
        elif pos[1] >= self.bounds[1, 1]:
            delta_x = pos[1] - self.bounds[1, 1]
            new_pos[1] = self.bounds[1, 0] + delta_x
        return new_pos
    
    def _get_value(self): 
        pos = self.current_position
        # if the agent is inside the data grid, 
        # returns the data point in its current position
        # otherwise, it returns 0
        data = self.data[self.t : self.t + 1, pos[0], pos[1]] if self._is_in_domain(self.current_position) else np.zeros(1, dtype=np.float64)
        self.t += 1
        if self.t >= self.MAX_T:
            self.t = 0        
        return data

                    
    def play_action(self, action_idx : int, clip_position : bool = True):
        self.current_position = self.current_position + self.velocity * COORDS[action_idx]
        if clip_position:
            self.current_position = self.clip_position(self.current_position)
        terminated = self.source_reached()
        reward = self.rewards[0] if terminated else self.rewards[1]
        return self._get_value(), reward, terminated

        
    def _get_values(self, num_values):
        pos = self.current_position
        if self._is_in_domain(pos):
            if self.t + num_values > self.MAX_T:
                delta = self.t + num_values - self.MAX_T
                d1 = self.data[self.t :self.MAX_T, pos[0], pos[1]]
                d2 = self.data[: delta, pos[0], pos[1]]
                data = np.concatenate((d1, d2))
                self.t = delta
            else:
                data = self.data[self.t : self.t + num_values, pos[0], pos[1]]
                self.t = self.t + num_values
        else:
            data = np.full(num_values, 0.0, dtype=np.float64) # if agent is out of bounds, return zeros
            self.t = (self.t + num_values) % self.MAX_T
        return data

        
    def reset(self, options : Dict):
        ## options = {
        #   init_position : np.array([y, x]),
        #   init_time : t (integer >= 0),
        #   sample_size : delta (integer > 0)
        #}
        self.init_position = options['init_position'] if 'init_position' in options else self.rnd_state.randint(self.bounds[:, 0], self.bounds[:, 1]) 
        self.current_position = self.init_position.copy()
        self.t = options['init_time'] if 'init_time' in options else self.rnd_state.randint(0, self.MAX_T, size=1)
        sample_size = options['sample_size'] if 'sample_size' in options else 1
        data = self._get_values(sample_size)
        return data
        
        