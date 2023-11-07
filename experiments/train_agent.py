import os
import time
import numpy as np

import matplotlib.pyplot as plt

from pynav.agents import BCKSMTAgent
from pynav.environment import Environment
from pynav.q_functions import TabularQ
from pynav.fmap import FMap
from pynav.stepsize import StepSize

from math import isfinite

data = np.load("/storage/rando/data/nose_data.npy")
n_thr = 3e-6 

data[data < n_thr] = 0.0

data = np.swapaxes(data, 0, 2)

thresholds = [
    np.array([0.1, 0.3, 0.6]), # intermittency
    np.zeros(4), # intensity
    np.array([0.5])
]

fmap = FMap(n_thr = n_thr, thresholds=thresholds)

env = Environment(
    data = data,
    n_thr = n_thr,
    init_position = np.array([0, 0, 0]),
    terminal_position = np.array([20, 142]),
    velocity = 10,
)

agent = BCKSMTAgent(
    env = env,
    q_function = TabularQ((4, 5, 2, 4), init_value=0.0),
    fmap = fmap,
    memory = 20,
    gamma = 0.99999,
    epsilon = StepSize(mode = 'sqrt', init_value=0.8, decay=1e-1), 
    alpha   = StepSize(mode = 'linear', init_value=0.9, decay=1e-1), 
)

H = 1000
n_episodes = 50001

rnd_state = np.random.RandomState(seed = 121314)

out_path = "./results/bck_agent/training"
os.makedirs(out_path, exist_ok=True)
values = []
#agent.load_Q("./q_function.npy")
with open(f"{out_path}/training.log", 'a') as f:
    for t in range(n_episodes):
        x = rnd_state.randint(100, high = 150, size=1)[0]
        y = rnd_state.randint(140, high = 160, size=1)[0]
        agent.env.init_position = np.array([x, y])
        episode_time = time.time()
        steps, total_reward, total_value, path, n_zeros, terminated = agent.step(horizon = H, callback = None, clip_position = False, test=False) 
        episode_time = time.time() - episode_time
        values.append(total_value)
        print("[{}/{}] x = {}\ty = {}\tvalue = {}\ttime = {}s\teps = {}\talpha = {}".format(t, n_episodes, x, y, round(total_value,4), round(episode_time, 4), round(agent.epsilon(), 4), round(agent.alpha(), 4)))
        f.write("{},{},{},{},{}\n".format(x, y, total_reward, steps, total_value))
        f.flush()
        if t > 0 and t % 10000 == 0:
            agent.store_Q(f"{out_path}/Q_{t}")
        
with open(f"{out_path}/training.log", 'r') as f:
    lines = f.readlines()

values = [float(line.split(",")[-1]) for line in lines]
            
agent.store_Q(f"{out_path}/q_function")
    
values = np.array(values)
    
delta = 500
avg_rewards = []
fig, ax = plt.subplots()
for i in range(values.shape[0] - delta):
    avg_rewards.append(np.mean(values[i:i+delta]))
    
ax.plot(range(len(avg_rewards)), avg_rewards, '-', c='black')
ax.set_xlabel("episodes")
ax.set_ylabel("average value")
fig.savefig(f"{out_path}/cumulative_average_reward.png", bbox_inches='tight')
    
