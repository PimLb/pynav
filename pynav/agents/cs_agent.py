import numpy as np



class CSAgent:
    
    def __init__(self, 
                 s_thr : float = 3e-6,
                 seed : int = 12131415,
                ):
        self.s_thr = s_thr
        self.k = 0
        self.Q_used = 0
        self.num_episodes = 0
        self.rnd_state = np.random.RandomState(seed)
        self.current_state = 0
                

    def update_state(self, raw_obs):
        if raw_obs == -1.0:
            return -1.0
        if raw_obs >= self.s_thr:
            self.current_counter = 0
            self.current_max = 1
            self.current_state = 0
        else:
            self.current_state = 1
            self.current_blank += 1
            self.current_state  = self.current_blank
        return self.current_state

#    def select_action(self, deterministic = False):

              
    def reset(self):
        self.current_counter = 0
        self.current_max = 1

