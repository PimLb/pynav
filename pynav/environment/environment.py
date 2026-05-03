import numpy as np




class Environment:
    
    def __init__(self, 
                 data : np.ndarray, 
                 source_position : np.ndarray,
                 gamma : float, # discount factor
                 source_radius : float = 2, # source radius
                 seed : int = 1212121314,
                 ):
        self.data = data
        self.gamma = gamma
        self.source_position = source_position
        self.source_radius = source_radius
        self.rnd_state = np.random.RandomState(seed)
        self.MAX_T = self.data.shape[0] 
                
    def _is_in_domain(self, pos):
        return 0 <= pos[0] < self.data.shape[1] and 0 <= pos[1] < self.data.shape[2]

    def source_reached(self, pos):
        return np.sum((pos - self.source_position) ** 2) <= (self.source_radius ** 2)

    def get_distance_to_source(self, pos):
        manhattan_distance = np.sum(np.abs(pos - self.source_position))    
        distance_to_border = max(0, manhattan_distance - self.source_radius)
        return np.ceil(distance_to_border)

    def compute_tb(self, pos, s_thr):
        if not self._is_in_domain(pos):
            return np.inf
        whiffs = self.data[:, pos[0], pos[1]] > s_thr
        idx = np.argmax(whiffs)
        blanks = []
        if whiffs[idx] == 0:
            return np.inf
        tb = 0
        for i in range(idx, whiffs.shape[0]):
            if not whiffs[i]:
                tb += 1
            else:
                if tb > 0:
                    blanks.append(tb)
                tb = 0
        return np.mean(blanks) if len(blanks) > 0 else 0.0
        
    def get_observation(self, pos, t): 
        # if the agent is inside the data grid, 
        # returns the data point in its current position
        # otherwise, it returns 0
        if self._is_in_domain(pos):
            return self.data[t : t + 1, pos[0], pos[1]]
        return np.zeros(1)

                    
    def make_move(self, pos, action : np.ndarray):
        pos = pos + action
        terminated = self.source_reached(pos)
        reward = 1.0 if terminated else - (1 - self.gamma)
        return pos, reward, terminated

        
        