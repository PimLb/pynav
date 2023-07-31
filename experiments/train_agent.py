import sys
import numpy as np

from qnav import Agent, Environment, TabularQ, FMap, BacktrackingAgent, AdaSensing
from qnav.stepsize import StepSize
from utils import train_agent, get_preprocessed_data, write_info


agent_type = sys.argv[1]
memory_size = int(sys.argv[2])
velocity = int(sys.argv[3])
n_thr = float(sys.argv[4])

num_episodes = int(sys.argv[5])
horizon = int(sys.argv[6])
out_path = f"./results/agent_{agent_type}/{memory_size}_{velocity}_{n_thr}/{num_episodes}_{horizon}"

source_radius = (20, 20)
gamma = 0.99999


data_path = "../../data/nose_data.mat"
data = get_preprocessed_data(data_path, data_key = 'nose', noise_thr = n_thr, rem=True)

terminal_position = np.asarray([140, 20], dtype=np.int32)

env = Environment(
        data, 
        init_position = np.asarray([0, 0]), 
        terminal_position = terminal_position,
        source_radius = source_radius,
        velocity = velocity,
        bounds=np.asarray([[-10000, 10000],[-5000, 8000]]),
        n_thr = n_thr,
        seed = 121212
    )

intermittency = np.asarray([0.33, 0.66])

thresholds = [intermittency, np.zeros(4)]

fmap = FMap(thresholds)

qfun = TabularQ((3, 5, 4))

gen_bounds = np.asarray([[0, 280], [40, 1225]], dtype=np.int32).reshape(-1, 2)

agent = Agent(env = env, 
              fmap = fmap, 
              qfun= qfun, 
              memory_size = memory_size,
              gamma = gamma,
              alpha = StepSize(mode = 'sqrt', init_value = 0.25, decay = 1e-2), 
              epsilon = StepSize(mode = 'sqrt', init_value = 0.99, decay = 1e-2),
              )

write_info(agent, num_episodes, horizon, out_path)

train_agent(agent, num_episodes, horizon, gen_bounds, out_path, store_each = 5000)