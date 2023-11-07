import numpy as np

from pynav.agents.q_agent import QAgent

class SMTAgent(QAgent):
    def __init__(self, seed=12, rnd_on_zero = True, **kwargs):
        super().__init__(**kwargs)
        self.first_obs = True
        self.rnd_on_zero = rnd_on_zero
        self.rnd_state = np.random.RandomState(seed)
                
    # def sensitivity_thr(self, raw_obs):        
    #     return np.max([0.5 * self.stat_memory.mean(), self.noise_thr])
                
    def _update_sensing_memory(self, raw_obs):
        if self.first_obs:
            self.stat_memory = raw_obs.copy().reshape(-1)
            self.first_obs = False
        else:
            self.stat_memory = np.delete(self.stat_memory, 0)
            self.stat_memory = np.concatenate((self.stat_memory, raw_obs.copy())).reshape(-1)


    def get_state(self, raw_obs, action, prev_features):
        self._update_sensing_memory(raw_obs)            
#        s_thr = self.sensitivity_thr(raw_obs)
        features = self.process_observation(self.stat_memory, action = action, prev_features = prev_features)
        self.fmap.update(raw_obs, features)
        f_idx = self.fmap.to_index(list(features.values()))
        return f_idx, features#, s_thr

    def get_action(self, raw_obs, state, features, test, action, s_thr):
        # if self.rnd_on_zero and test and len(self.stat_memory[self.stat_memory >= s_thr]) == 0:
        #     return self.rnd_state.randint(0, 4)
        if len(self.stat_memory[self.stat_memory >= self.noise_thr]) == 0:
            return self.rnd_state.randint(0, 4)
        return self.pi.sample_epsilon_greedy(state, self.epsilon())

    def reset(self):
        self.stat_memory = np.zeros(self.memory, dtype=np.float64)
        self.first_obs = True