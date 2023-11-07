import numpy as np



from pynav import QAgent
from pynav.environment import Environment
from pynav.feature_maps import FeatureMap, TDFeatureMap

from pynav.utils import StepSize, DeltaRelativeError

import matplotlib.pyplot as plt



data = np.load("nose_data_35_153.npy")
data[data < 3e-6] = 0.0

def s_thr(memory):
    return 3e-6

source_position = np.array([17, 2])
source_radius = (20, 20)
velocity = 10
fmap = TDFeatureMap(interm_disc=np.array([0.33, 0.66]), time_disc=np.array([i for i in range(1, 10)]))
sampling_regions = [
    np.array([[0, data.shape[1]], [90, 100]])# data.shape[2]]])
]
T = 200
gamma = 0.999#np.exp(-lam * T)

env = Environment(data = data, domain=None, sampling_regions=sampling_regions,  source_position=source_position, source_radius=source_radius, velocity=velocity, boundary_conditions=False, random_time=True)
agent = QAgent(env=env, 
               gamma=gamma, 
               fmap=fmap, 
               s_thr=s_thr, 
               memory_size=10, 
               init_Q = 0.0, 
               lr = StepSize(0.50, decay='sqrt', decay_constant=0.01, min_value=0.01), 
               eps=StepSize(0.90, decay='sqrt', decay_constant=0.01),
#               convergence_crit=DeltaRelativeError(1e-4, 2)
               )


int_levels = {
    0: 'low',
    1: 'mid-low',
    2: 'mid',
    3: 'mid-high',
    4: 'high'
}

iterm_levels = {
    0 : 'low',
    1 : 'mid',
    2 : 'high'
}

default_action_map = {
    0 : 'up', #np.array([1.0, 0.0]),
    1 : 'down',#np.array([-1.0, 0.0]),
    2 : 'right',#np.array([0.0, 1.0]),
    3 : 'left'#np.array([0.0, -1.0])
}


agent.load("./q_function.npy")

print(agent.Q.shape)

for i in range(agent.Q.shape[0]):
    for j in range(agent.Q.shape[1]):
        print("intermittency = {}\tintensity = {} => {}".format(iterm_levels[i], int_levels[j], default_action_map[np.argmax(agent.Q[i, j])] ))