import numpy as np

from typing import Dict, List

default_actions = {
    0 : np.asarray([0, 1], dtype=np.int32),
    1 : np.asarray([0, -1], dtype=np.int32),
    2 : np.asarray([1, 0], dtype=np.int32),
    3 : np.asarray([-1, 0], dtype=np.int32),
}

default_inverse = [1, 0, 3, 2]

class Environment:
    
    def __init__(self, 
                 data : np.ndarray, 
                 init_position : np.ndarray,
                 terminal_position : np.ndarray,
                 actions : Dict[int, np.ndarray] = default_actions,
                 inverse_actions : List = default_inverse,
                 velocity : int = 10,
                 source_radius : int = (20, 20),
                 n_thr : float = 3e-6,
                 bounds : np.ndarray = None,
                 seed : int = 1212
                 ):
        self.data = data # data.shape = (TIMESTEPS, Ys, Xs)
        self.actions = actions
        self.inverse_actions = inverse_actions
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed)
        self.bounds = bounds # [[min_y, max_y], [min_x, max_x]]
        self.velocity = velocity
        self.n_thr = n_thr
        if bounds is None:
            self.bounds = np.asarray([
                [0, self.data.shape[1]], 
                [0, self.data.shape[2]],
            ])
        self.init_position = init_position.copy()
        self.current_position = init_position.copy()
        self.terminal_position = terminal_position
        self.source_radius = source_radius
        self.t = 0
        self.MAX_T = self.data.shape[0] 
                
    def _is_in_bounds(self, pos):
        # pos = [y, x]
        return self.bounds[0, 0] <= pos[0] < self.bounds[0, 1] and self.bounds[1, 0] <= pos[1] < self.bounds[1, 1]

    def _is_in_discretization(self, pos):
        return 0 <= pos[0] < self.data.shape[1] and 0 <= pos[1] < self.data.shape[2]

    def _is_near_enough(self, pos):
        return (abs(pos[0] - self.terminal_position[0]) <= self.source_radius[0]) and abs(pos[1] - self.terminal_position[1]) <= self.source_radius[1]
        
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


    def _get_observation(self): 
        '''get a raw observation in current position'''
        pos = self.current_position
        if self._is_in_discretization(self.current_position):
            data = self.data[self.t : self.t + 1, pos[0], pos[1]]
        else: # if agent is out of bounds, return zeros
            data = np.zeros(1, dtype=np.float64) 
        self.t += 1
        if self.t >= self.MAX_T:
            self.t = 0        
        return data
            

    def step(self, action_idx, clip_position : bool = False): 
        v = self.actions[action_idx]
        # Update position
        self.current_position = self.current_position + self.velocity * v
        if clip_position:
            self.current_position = self.clip_position(self.current_position)        
        # Return raw observation, reward and check if the agent is in the source region
        raw_obs = self._get_observation() 
        terminated = self._is_near_enough(self.current_position)
        reward = 1.0 if terminated else -0.001
        return raw_obs, reward, terminated

        
        
    def reset(self, num_obs : int):
        self.current_position = self.init_position.copy()
        self.t = num_obs
        if not self._is_in_discretization(self.current_position):
            return np.zeros(num_obs, dtype=np.float64), False
        data = self.data[:num_obs, self.current_position[0], self.current_position[1]]
        return data, self._is_near_enough(self.current_position)
        
        