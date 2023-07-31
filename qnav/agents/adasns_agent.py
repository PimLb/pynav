import numpy as np

from qnav.agents.agent import Agent
from qnav.fmap.fmap import FMap
from qnav.environment.env import Environment
from qnav.policy.policy import ImplicitPolicy
from qnav.qfunctions.qfunctions import TabularQ
from qnav.stepsize.stepsize import StepSize

from qnav.utils.utils import estimate_tb_duration

class AdaSensing(Agent):
    
    def __init__(self, init_memory = None, min_memory = 5, tb_const = 2,**kwargs):
        super().__init__(**kwargs)
        self.init_memory = init_memory
        self.current_memory = init_memory
        self.tb_const = tb_const
        self.current_tb = 0
        self.open_tb = False
        self.min_memory = min_memory

    @property
    def sensitivity_threshold(self):
        return max(self.s_thr_const * np.mean(self.memory[-self.current_memory:]), self.n_thr)

    def process_observation(self, s_thr, action = None, prev_features = None):
        curr_mem = self.memory[-self.current_memory:]
        whiff = curr_mem >= s_thr
        intermittency = np.sum(whiff) / curr_mem.shape[0]
        avg_intensity_during_whiff = curr_mem[whiff].mean() if np.any(whiff) else 0.0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
        }
        return features

    def get_action(self, state, features, s_thr, test : bool = False, action = None):
        curr_mem = self.memory[-self.current_memory:]
        memory_is_empty = len(curr_mem[curr_mem >= s_thr]) == 0
        if test and self.rnd_on_zero and memory_is_empty:
            return self.rnd_state.randint(0, len(self.actions.keys()))
        return self.pi.sample_epsilon_greedy(state, self.epsilon())

    def memory_is_empty(self, s_thr):
        curr_mem = self.memory[-self.current_memory:]
        return len(curr_mem[curr_mem >= s_thr]) == 0
    
    def initialize_memory(self):
        if self.current_memory is None:
            s_thr = max(self.s_thr_const * np.mean(self.memory), self.n_thr)
            tb_duration, indices = estimate_tb_duration(self.memory, s_thr)
            self.current_memory = max(tb_duration) + self.tb_const if len(tb_duration) > 0 else self.min_memory
            if len(indices) > 0:
                self.open_tb = True
                self.current_tb = self.memory.shape[0] - indices[-1]
        
    def get_state(self, raw_obs, prev_action, prev_features):
        self.update_memory(raw_obs)
        s_thr = self.sensitivity_threshold
        if self.open_tb:
            if raw_obs < s_thr:
                self.current_tb += 1
            else:
                self.current_memory = self.current_tb + self.tb_const
                self.current_memory = self.current_memory if self.current_memory > self.min_memory else self.min_memory
                self.open_tb = False
                self.current_tb = 0
        else:
            prev, curr = self.memory[-2], self.memory[-1]
            if prev >= s_thr and curr < s_thr:
                self.open_tb = True
                self.current_tb += 1 
        features = self.process_observation(s_thr, action = prev_action, prev_features = prev_features)
        self.fmap.update(raw_obs, features, s_thr)
        f_idx = self.fmap.to_index(list(features.values()))
        return f_idx, features, s_thr
    
    def reset(self):
        super().reset()
        self.current_memory = self.init_memory        
        self.current_tb = 0
        self.open_tb = False
