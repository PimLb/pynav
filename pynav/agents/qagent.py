import os
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
                 conv_params = (100, 1e-5),
                 s_thr : float = 3e-6,
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
        self.conv_params = conv_params
        self.out_dir = out_dir
        self.num_episode = 0
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
        obs = int(raw_obs > self.s_thr)
        self.memory = np.concatenate((self.memory[1:], [obs]))
        if np.all(self.memory[-self.memory_length:] == 0.0):        
            self.current_state = self.current_state + 1
            if self.current_state >= self.time_discretization_size:
                self.current_state = 1
        else:
            self.current_state = 0
        return self.current_state

    def select_action(self):
        if self.rnd_state.rand() < self.eps_greedy(self.num_episode):
            return self.rnd_state.choice(self.number_of_actions, 1)[0]
        return np.argmax(self.Q[self.current_state])

    def update_q(self, trace):
        # trace =[(s, a, s', r)]
        alpha = self.learning_rate(self.num_episode)
        for (s, a, s_prime, r) in trace:
            self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + self.gamma * np.max(self.Q[s_prime]))
        self.num_episode += 1
          
    def reset(self):
        self.memory = np.zeros(self.memory_size)
        self.current_state = 0
        