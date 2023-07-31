import os
import numpy as np
import h5py
import time 


class Result:
    
    def __init__(self, out_dir, store_memstate = False, store_fmap = False):
        self.out_dir = out_dir
        self.position_paths = []
        self.fmap_int_states = []
        self.fmap_thr_states = []
        self.rewards = []
        self.discounted_rewards = []
        self.converged = []
        self.steps = []
        self.int_path = []
        self.mem_states = []
        self.num_zeros = []
        self.store_memstate = store_memstate
        self.store_fmap = store_fmap

    def _write_positions(self):
        with open(f"{self.out_dir}/position_path", 'a') as f:
            for (i, path) in enumerate(self.position_paths):
                for (j, pos) in enumerate(path):
                    intensity = self.int_path[i][j] if j < len(self.int_path[i]) else self.int_path[i][-1]
                    f.write(f"{pos[0]},{pos[1]},{intensity}\n")          
                f.write("-"*55 + "\n")

    def _write_state_path(self, fname, state_paths):
        with open(f"{self.out_dir}/{fname}", 'a') as f:
            for path in state_paths:
                for elem in path:
                    f.write(','.join([str(x) for x in elem]) + "\n")          
                f.write("-"*55 + "\n")

    def _write_rewards(self):
        with open(f"{self.out_dir}/rewards", 'a') as f:
            for i in range(len(self.position_paths)):
                init_pos = self.position_paths[i][0]
                f.write(f"{init_pos[0]},{init_pos[1]},{self.rewards[i]},{self.discounted_rewards[i]},{self.steps[i]},{self.converged[i]},{self.num_zeros[i]}\n")

    def write_results(self):
        self._write_positions()
        self._write_rewards()
        if self.store_memstate:
            self._write_state_path("mem_states", self.mem_states)
        if self.store_fmap:
            self._write_state_path("fmap_thresholds", self.fmap_thr_states)
            self._write_state_path("fmap_intensities", self.fmap_int_states)

    def add_result(self, result):
        self.position_paths.append(result['position_path'])
        if self.store_fmap:
            self.fmap_int_states.append(result['fmap_state'][0])
            self.fmap_thr_states.append(result['fmap_state'][1])
        self.rewards.append(result['reward'][0])
        self.discounted_rewards.append(result['reward'][1])
        self.converged.append(result['terminated'])
        self.steps.append(result['steps'])
        self.int_path.append(result['int_path'])
        if self.store_memstate:
            self.mem_states.append(result['mem_state'])
        self.num_zeros.append(result['num_zeros'])

def write_info(agent, num_episodes, horizon, out_path):
    os.makedirs(f"{out_path}", exist_ok=True)
    with open(f"{out_path}/info", 'w') as f:
        f.write(f"[--] num episodes: {num_episodes}\thorizon: {horizon}\n")
        f.write(f"[--] velocity: {agent.env.velocity}\tmem size: {agent.memory_size}\n")
        f.write(f"[--] gamma: {agent.gamma}\n")
        f.write(f"[--] noise thr constant: {agent.s_thr_const}\n")
        f.write(f"[--] alpha: {agent.alpha}\n")
        f.write(f"[--] epsilon: {agent.epsilon}\n")
        f.write(f"[--] agent seed: {agent.seed}\tenv seed: {agent.env.seed}\n")

        
def read_data(data_path, data_key = 'nose'):
    print("[--] Reading data and coordinates...")
    f = h5py.File(data_path)
    reading_time = time.time()
    data = np.asarray(f[data_key])
    reading_time = time.time() - reading_time
    print("[--] Data read in {}s!".format(round(reading_time, 4)))
    f.close()
    return data

def get_preprocessed_data(data_path, data_key = 'nose', noise_thr = 3e-6, rem=True):
    data = read_data(data_path=data_path, data_key=data_key)
    p_time = time.time()
    if rem:
        data = np.delete(data, 1732, axis=0)
        data = np.delete(data, 2460, axis=0) 
    data[data <= noise_thr] = 0.0
    p_time = time.time() - p_time
    print("[--] Data preprocessed in {}s!".format(round(p_time, 4)))
    return data

def train_agent(agent, num_episodes, horizon, gen_bounds, out_path, store_each = 1000):
    os.makedirs(f"{out_path}/training", exist_ok=True)
    os.makedirs(f"{out_path}/q_functions", exist_ok=True)
    rnd_state = np.random.RandomState(12131415)
    for i in range(num_episodes):
        result = Result(f"{out_path}/training", store_memstate = False, store_fmap = False)
        init_y, init_x = rnd_state.randint(gen_bounds[0, 0], gen_bounds[0, 1]), rnd_state.randint(gen_bounds[1, 0], gen_bounds[1, 1])
        agent.set_init_position(init_x, init_y)
        agent.reset()
        episode_time = time.time()
        episode_result = agent.run_episode(horizon = horizon, clip_position = True, test = False)
        episode_time = time.time() - episode_time
        agent.update_lr()
        print(f"[{i}/{num_episodes}] x: {init_x}\ty: {init_y}\tvalue: {round(episode_result['reward'][1], 4)}\tconverged: {episode_result['terminated']}\ttime: {round(episode_time, 4)}")
        if i > 0 and i % store_each == 0:
            agent.store_Q(f"{out_path}/q_functions/QFunction_{i}")
        result.add_result(episode_result)
        result.write_results()
    agent.store_Q(f"{out_path}/QFunction")
    