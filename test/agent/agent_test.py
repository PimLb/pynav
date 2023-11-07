import numpy as np

from pynav import QAgent, BacktrackingAgent
from pynav.environment import Environment
from pynav.feature_maps import FeatureMap
import matplotlib.pyplot as plt

data = np.zeros((100, 500, 500))#np.load("/storage/rando/data/nose_data_35_153.npy")
data[data < 3e-6] = 0.0
fmap = FeatureMap(interm_disc=[0.33, 0.66])
source_position=(17, 2)

data[:5, 0, 200] = 1.0
env = Environment(data = data, source_position=source_position)

agent = BacktrackingAgent(env=env, fmap=fmap, memory_size=5)

observations, _ = agent.env.reset(params=dict(num_obs=6, init_position=np.array([0, 200, 0])))
print("DATA IN POS: ", data[:5, 20, 40], "observations: ", observations)

agent.Q[:, :, 3] = 1.0

agent.initialize(observations)

print(agent.env.position)
act = None
for i in range(10):
    a = None
    obs, reward, terminated, info, a = agent.step(a=act)
    print(agent.env.position, terminated, a)
    if terminated:
        break
    agent.update(obs)
    print(agent.memory, agent.backtracking_stack)
    print(agent.fmap.avgint_disc)
    print("-"*44)
    act = a
