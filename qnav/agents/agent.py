import numpy as np

from typing import Dict

from qnav.fmap.fmap import FMap
from qnav.environment.env import Environment
from qnav.policy.policy import ImplicitPolicy
from qnav.qfunctions.qfunctions import TabularQ
from qnav.stepsize.stepsize import StepSize


class Agent:
    
    def __init__(self, 
                 env : Environment,
                 fmap : FMap,
                 qfun : TabularQ,
                 memory_size : int = 1,
                 gamma : float = 0.99,
                 s_thr_const : float = 0.5,
                 alpha : StepSize = StepSize(mode = 'sqrt', init_value = 0.25, decay = 1e-2), 
                 epsilon : StepSize = StepSize(mode = 'sqrt', init_value = 0.99, decay = 1e-2),
                 seed : int = 12131415,
                 rnd_on_zero : bool = True
                 ):
        self.env = env
        self.fmap = fmap
        self.qfun = qfun
        self.gamma = gamma
        self.pi = ImplicitPolicy(qfun, seed = seed)
        self.actions = env.actions
        self.inverse_actions = env.inverse_actions
        self.memory_size = memory_size
        self.memory = np.zeros(memory_size, dtype=np.float64)
        self.num_episodes = 0
        self.alpha = alpha
        self.epsilon = epsilon
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed = seed)
        self.s_thr_const = s_thr_const
        self.rnd_on_zero = rnd_on_zero
        self.current_memory = memory_size
        
    @property
    def n_thr(self):
        return self.env.n_thr
        
    @property
    def sensitivity_threshold(self):
        return max(self.s_thr_const * np.mean(self.memory), self.n_thr)
        
    def store_Q(self, file : str):
        np.save(file, self.qfun.data, allow_pickle=False)

    def load_Q(self, file : str):
        self.qfun.data = np.load(file, allow_pickle=False)
        
    def update_Q(self, reward : float, state : int, action : int, next_state : int, next_action : int):
        alpha = self.alpha()
        delta = reward + self.gamma * self.qfun.of(next_state, next_action)
        self.qfun.update(state, action, delta, alpha)
    
    def set_init_position(self, init_x, init_y):
        self.env.init_position[0] = init_y
        self.env.init_position[1] = init_x
           
    def update_memory(self, raw_obs):
        self.memory = np.delete(self.memory, range(raw_obs.shape[0]))
        self.memory = np.concatenate((self.memory, raw_obs.copy())).reshape(-1)
               
    def process_observation(self, s_thr, action = None, prev_features = None):
        whiff = self.memory >= s_thr
        intermittency = np.sum(whiff) / self.memory.shape[0]
        avg_intensity_during_whiff = self.memory[whiff].mean() if np.any(whiff) else 0.0
        features = {
            'intermittency' : intermittency, 
            'avg_int' : avg_intensity_during_whiff,
        }
        return features
           
    def get_state(self, raw_obs, prev_action, prev_features):
        self.update_memory(raw_obs)
        s_thr = self.sensitivity_threshold
        features = self.process_observation(s_thr, action = prev_action, prev_features = prev_features)
        self.fmap.update(raw_obs, features, s_thr)
        f_idx = self.fmap.to_index(list(features.values()))
        return f_idx, features, s_thr
    
    def memory_is_empty(self, s_thr):
        return len(self.memory[self.memory >= s_thr]) == 0

    def get_action(self, state, features, s_thr, test : bool = False, action = None):
        if test and self.rnd_on_zero and self.memory_is_empty(s_thr): 
            return self.rnd_state.randint(0, len(self.actions.keys()))
        return self.pi.sample_epsilon_greedy(state, self.epsilon())
        
    def step(self, raw_obs, prev_action = None, prev_features = None, clip_position : bool = True, test = False):
        s, phi_s, s_thr = self.get_state(raw_obs, prev_action, prev_features)
        a = self.get_action(s, phi_s, s_thr, test = test, action = prev_action)
        new_obs, reward, terminated = self.env.step(a, clip_position = clip_position)
        return s, phi_s, s_thr, a, new_obs, reward, terminated            

    def initialize_memory(self):
        self.current_memory = self.memory_size


    def run_episode(self, horizon = None, clip_position : bool = True, test : bool = False):
        raw_obs, terminated = self.env.reset(self.memory_size)
        position_path = [self.env.init_position.copy()]
        int_path = []
        mem_state = [self.memory.copy()]
        fmap_int_state = [self.fmap.intensities.copy()]
        fmap_thr_state = [self.fmap.thresholds[1].copy()]
        total_reward, discounted_reward = 0.0, 0.0
        num_zeros = 0 # number of zeros in memory
        k = 0
        prev_action, prev_phi = None, None
        self.initialize_memory()
        mem_used = [self.current_memory]
        while not terminated and (horizon is None or k < horizon):
            s, phi_s, s_thr, a, new_obs, reward, terminated = self.step(raw_obs, prev_action=prev_action, prev_features = prev_phi, clip_position = clip_position, test = test)
            if not test:
                phi_prime = self.process_observation(s_thr, action = a, prev_features = phi_s)
                s_prime = self.fmap.to_index(list(phi_prime.values()))
                a_prime = self.pi.sample_greedy(s_prime)
                self.update_Q(reward, s, a, s_prime, a_prime)
            prev_action = a
            prev_phi = phi_s
            raw_obs = new_obs
            total_reward += reward
            discounted_reward += (self.gamma ** k) * reward
            if self.memory_is_empty(s_thr):#len(self.memory[self.memory >= s_thr]) == 0:
                num_zeros += 1
            int_path.append(phi_s['avg_int'])            
            mem_state.append(self.memory.copy())
            mem_used.append(self.current_memory)
            position_path.append(self.env.current_position.copy())            
            fmap_int_state.append(self.fmap.intensities.copy())
            fmap_thr_state.append(self.fmap.thresholds[1].copy())
            k += 1
        result = {
            'reward' : (total_reward, discounted_reward),
            'terminated' : terminated,
            'steps' : k,
            'mem_used' : mem_used,
            'num_zeros' : num_zeros,
            'int_path' : int_path,
            'mem_state' : mem_state,
            'position_path' : position_path,
            'fmap_state' : (fmap_int_state, fmap_thr_state)            
        }
        return result 

    def update_lr(self):
        self.num_episodes += 1
        self.alpha.next()
        self.epsilon.next()

    def reset(self):
        self.memory = np.zeros(self.memory_size, dtype=np.float64)
        self.fmap.reset()
        self.current_memory = self.memory_size

        
        