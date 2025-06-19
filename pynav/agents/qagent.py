import numpy as np



class QAgent:
    
    def __init__(self, 
                 horizon,
                 eps_greedy = lambda t : 0.3,
                 gamma : float = 0.99,
                 learning_rate = lambda t : 0.25,
                 s_thr : float = 3e-6,
                 seed : int = 12131415,
                 out_dir : str = "./"
                ):
        self.s_thr = s_thr
        self.gamma = gamma
        self.horizon = horizon
        self.Q = np.full((horizon + 2, 4), -1.0, dtype=np.float32)
        self.out_dir = out_dir
        self.k = 0
        self.Q_used = 0
        self.num_episodes = 0
        self.rnd_state = np.random.RandomState(seed)
        self.learning_rate = learning_rate
        self.eps_greedy = eps_greedy
        self.current_state = 0
        self.current_blank = 0
                
    def store_q(self, fname):
        np.save("{}/{}".format(self.out_dir, fname), self.Q, allow_pickle=True)

    def load_q(self, qpath):
        self.Q = np.load(qpath, allow_pickle=True)

    def update_state(self, raw_obs):
        if raw_obs == -1.0:
            return -1.0
        if raw_obs >= self.s_thr:
            self.current_blank = 0
            self.current_state = 0
        else:
            self.current_blank += 1
            self.current_state  = self.current_blank
        return self.current_state

    def select_action(self, deterministic = False):
        if not deterministic and self.rnd_state.rand() < self.eps_greedy(self.num_episodes):
            return self.rnd_state.choice(self.Q.shape[1], 1)[0]
        return np.argmax(self.Q[self.current_state])

    def update_q(self, s, a, s_prime, r):
        alpha = self.learning_rate(self.num_episodes)
        maxQ = np.max(self.Q[s_prime]) if s_prime != -1.0 else 0.0
        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + self.gamma * maxQ)
          
              
    def reset(self):
        self.current_blank = 0
        self.current_state = 0
        self.Q_used = 0



class TwoQAgent(QAgent):
    
    def __init__(self, horizon, tau=1, eps_greedy=lambda t: 0.3, gamma: float = 0.99, learning_rate=lambda t: 0.25, s_thr: float = 0.000003, seed: int = 12131415, use_right_Q = False, out_dir: str = "./"):
        self.s_thr = s_thr
        self.gamma = gamma
        self.horizon = horizon
        self.Q = np.full((2*horizon + 4, 4), -1.0, dtype=np.float32)
        self.out_dir = out_dir
        self.k = 0
        self.Q_used = 0
        self.num_episodes = 0
        self.rnd_state = np.random.RandomState(seed)
        self.learning_rate = learning_rate
        self.eps_greedy = eps_greedy
        self.current_state = 0
        self.current_blank = 0
        self.tau = tau
        self.use_right_Q = use_right_Q
    
    
    def update_state(self, raw_obs):
        if raw_obs == -1.0:
            return -1.0
        if raw_obs >= self.s_thr:
    #        if self.current_blank > 0:
            if not self.use_right_Q:
                self.Q_used = int(self.current_blank > self.tau)
            self.current_state = 0 
            self.current_blank = 0
        else:
            self.current_blank += 1            
            self.current_state = self.current_blank if self.Q_used == 0 else self.current_blank + self.horizon + 2
        return self.current_state

    def select_action(self, deterministic = False):
        if not deterministic and self.rnd_state.rand() < self.eps_greedy(self.num_episodes):
            return self.rnd_state.choice(4, 1)[0]
        return np.argmax(self.Q[self.current_state])

              
    def reset(self):
        self.current_blank = 0
        self.Q_used = 1#int(self.rnd_state.rand() < 0.5)
        self.current_state = 0 #if self.Q_used else self.horizon + 2

