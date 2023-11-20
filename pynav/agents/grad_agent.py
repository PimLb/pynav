import numpy as np


class GDAgent:
    
    def __init__(self, data, source_location, source_size, use_sign = False):
        self.data = data
        self.source_location = source_location
        self.source_size=source_size
        self.t = 0
        self.use_sign = use_sign
        self.I = np.eye(2)
        
    def _is_in_source(self, pos):
        return abs(pos[0] - self.source_location[0]) <= self.source_size and abs(pos[1] - self.source_location[1]) <= self.source_size
        
    def _get_obs(self, pos):
        if 0 <= pos[0] < self.data.shape[1] and 0 <= pos[1] < self.data.shape[2]:
            return self.data[self.t, int(pos[0]), int(pos[1])]
        return 0.0
        
    def approx_grad(self, pos):
        fx = self._get_obs(pos)
        f_next = [] 
        g = np.zeros(2)
        for i in range(2):
            f_next = self._get_obs(pos + self.I[i])
            g += (f_next - fx) * self.I[i]
        if self.use_sign:
            return np.sign(g)
        return g
    
    def optimize(self, init_position, init_t, gamma = 1, horizon=100, verbose=False):
        self.t = init_t
        pos = init_position
        path = [pos.copy()]
        for k in range(horizon):
            g = self.approx_grad(pos)
            pos = pos + gamma * g
            if verbose:
                print("[{}/{}] agent position = {}".format(k, horizon, pos))
            path.append(pos.copy())
            if self._is_in_source(pos):
                print("[--] Converged in {} steps!".format(k))
                break
            self.t += 1
            if self.t >= self.data.shape[0]:
                self.t = 0
        return path
        
class MemGDAgent:
    
    def __init__(self, data, source_location, source_size, use_sign = False, mem_size = 5):
        self.data = data
        self.source_location = source_location
        self.source_size=source_size
        self.mem_size = mem_size
        self.memory = []
        self.t = 0
        self.use_sign = use_sign
        self.I = np.eye(2)
        
    def _is_in_source(self, pos):
        return abs(pos[0] - self.source_location[0]) <= self.source_size and abs(pos[1] - self.source_location[1]) <= self.source_size
        
    def _get_obs(self, pos):
        if 0 <= pos[0] < self.data.shape[1] and 0 <= pos[1] < self.data.shape[2]:
            return self.data[self.t, int(pos[0]), int(pos[1])]
        return 0.0
        
    def approx_grad(self, pos):
        obs = self._get_obs(pos)
        self.memory.append(obs)
        if len(self.memory) > self.mem_size:
            self.memory.pop(0)
        fx = np.mean(self.memory)
        f_next = [] 
        g = np.zeros(2)
        for i in range(2):
            obs_next = self._get_obs(pos + self.I[i])
            mu_next = self.memory.copy()
            mu_next.append(obs_next)
            g += (np.mean(mu_next) - fx) * self.I[i]
        if self.use_sign:
            return np.sign(g)
        return g
    
    def optimize(self, init_position, init_t, gamma = 1, horizon=100, verbose=False):
        self.t = init_t
        pos = init_position
        path = [pos.copy()]
        for k in range(horizon):
            g = self.approx_grad(pos)
            pos = pos + gamma * g
            if verbose:
                print("[{}/{}] agent position = {}".format(k, horizon, pos))
            path.append(pos.copy())
            if self._is_in_source(pos):
                print("[--] Converged in {} steps!".format(k))
                break
            self.t += 1
            if self.t >= self.data.shape[0]:
                self.t = 0
        return path
 