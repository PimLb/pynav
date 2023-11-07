import numpy as np

from math import ceil

from pynav.agents.smt_agent import SMTAgent
from pynav.utils import count_consecutive

class AdaInvEregAgent(SMTAgent):
    def __init__(self, min_mem = 5, init_mem = 10, seed=12, rnd_on_zero= True, **kwargs):
        super().__init__(seed, rnd_on_zero, **kwargs)
        self.first_obs = True
        self.tb_len = 0
        self.init_mem = init_mem
        self.last_blank = 0
        self.min_mem = min_mem
        self.curr_len = init_mem #self.memory
        self.rnd_state = np.random.RandomState(seed)
        self.mem_used = []

    def build_features(self, raw_obs, action = None, prev_features = None):
        whiff = raw_obs >= self.noise_thr
        intermittency = np.sum(whiff) / len(raw_obs)
        avg_intensity_during_whiff = raw_obs[whiff].mean() if np.any(whiff) else 0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
        }
        features['zeros'] = 0
        if prev_features is None and np.all(raw_obs < self.noise_thr):
            features['zeros'] = 1
        elif prev_features is not None and np.all(raw_obs < self.noise_thr):
            features['zeros'] = prev_features['zeros'] + 1
        return features

    def _init_used_memory(self):
        self.curr_len = self.init_mem

    def _set_clen(self):
        self.curr_len = self.last_blank  if self.last_blank > self.min_mem else self.min_mem

    def process_observation(self, raw_obs, action = None, prev_features = None):
        if prev_features is None:
            self._init_used_memory()
        else:
            if np.any(self.stat_memory[-self.curr_len:] > self.noise_thr):
                if raw_obs[-1] < self.noise_thr:
                    self.tb_len += 1
                else:
                    if self.tb_len > 0 and self.tb_len > self.curr_len:
                        self.last_blank = self.tb_len
                    elif self.tb_len > 0:
                        self.last_blank -= 1
                    self._set_clen()
                    self.tb_len = 0         
        return self.build_features(self.stat_memory[-self.curr_len:], action = action, prev_features = prev_features)
                
    def get_action(self, raw_obs, state, features, test, action):
        mem = self.stat_memory[-self.curr_len:]
        return self.pi.sample_epsilon_greedy(state, self.epsilon())
                     

    def reset(self):
        super().reset()
        self.tb_len = 0
        self.curr_len = self.init_mem #self.memory
        self.last_blank = 0