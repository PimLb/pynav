import tqdm
import numpy as np



class QAgent:
    
    def __init__(self, 
                 number_of_actions : int = 4,
                 time_discretization_size : int = 30,
                 memory_size : int = 1,
                 q_init_value : float = 0.0,
                 eps_greedy = lambda t : 0.3,
                 gamma : float = 0.99,
                 learning_rate = lambda t : 0.25,
                 s_thr : float = 3e-6,
                 adaptive_thr  = -np.inf, 
                 seed : int = 12131415,
                 out_dir : str = "./"
                ):
        self.s_thr = s_thr
        self.memory_size = memory_size
        self.gamma = gamma
        self.memory = np.zeros(memory_size)
        self.number_of_actions = number_of_actions
        self.time_discretization_size = time_discretization_size
        self.Q = np.full((time_discretization_size, number_of_actions), q_init_value, dtype=np.float32)
        self.adaptive_thr = adaptive_thr
        self.out_dir = out_dir
        self.k = 0
        self.num_episodes = 0
        self.rnd_state = np.random.RandomState(seed)
        self.learning_rate = learning_rate
        self.eps_greedy = eps_greedy
        self.current_state = 0
        self.memory_length = memory_size
        
    def set_memory(self, memory_length):
        self.memory_length = memory_length
        
    def store_q(self, fname):
        np.save("{}/{}".format(self.out_dir, fname), self.Q, allow_pickle=True)

    def load_q(self, qpath):
        self.Q = np.load(qpath, allow_pickle=True)

    def update_state(self, raw_obs):
        self.memory = np.concatenate((self.memory[1:], raw_obs))
        s_thr = max(self.adaptive_thr * self.memory[-self.memory_length:].mean(), self.s_thr)
        bin_memory = self.memory[-self.memory_length:] > s_thr
        if np.all(bin_memory == 0.0):        
            self.current_state = self.current_state + 1
            if self.current_state >= self.time_discretization_size:
                self.current_state = 1
        else:
            self.current_state = 0
        return self.current_state

    def select_action(self, deterministic = False):
        if not deterministic and self.rnd_state.rand() < self.eps_greedy(self.num_episodes):
            return self.rnd_state.choice(self.number_of_actions, 1)[0]
        return np.argmax(self.Q[self.current_state])

    def update_q(self, s, a, s_prime, r):
        alpha = self.learning_rate(self.num_episodes)
        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + self.gamma * np.max(self.Q[s_prime]))
          
          
    # def test(self, env, init_positions, verbose = False):
    #     results = {}
    #     iterator = tqdm(init_positions)
        
          
    def reset(self):
        self.memory = np.zeros(self.memory_size)
        self.current_state = 0
        self.memory_length = self.memory_size
