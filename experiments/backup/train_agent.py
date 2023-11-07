import numpy as np

from pynav import QAgent, BacktrackingAgent, QSeqAgent
from pynav.environment import Environment
from pynav.feature_maps import FeatureMap
from pynav.recovery_procedures import BacktrackingRecovery, QRecovery

from pynav.utils import StepSize, DeltaRelativeError

import matplotlib.pyplot as plt

import time

load_time = time.time()
data = np.load("/storage/rando/data/nose_data.npy")
load_time = time.time() - load_time
print("Data loaded in {} seconds".format(round(load_time,4)))

n_thr = 3e-6
data[data <= n_thr] = 0.0

#data[data > 0.0] = np.log10(data[data > 0.0])

#print(data.shape)
#data = data.mean(0)
#data[data!=data] = 0.0
#data = data.reshape(1, 280, 1225)

source_location = (142, 20)
source_radius = (15, 15)

velocity = 10
fmap = FeatureMap(interm_disc=np.array([0.33, 0.66]))
sampling_regions = [
#    np.array([[190,200], [1150, 1200]]),
    np.array([[100,120], [500, 600]]),
]
T = 1000#0
gamma = 0.9999#9999

def s_thr(memory):
    return n_thr

#domain = np.load("/storage/rando/data/domain_nose_3e-06.npy")

env = Environment(data = data, domain=None, 
                  sampling_regions=sampling_regions,  
                  source_position=source_location, 
                  source_radius=source_radius, 
                  rewards = (-0.01, 1.0),
                  velocity=velocity, 
                  boundary_conditions=True,
                  truncate=False, 
                  random_time=True)

agent = QSeqAgent(env=env, 
               gamma=gamma, 
               s_thr=3e-6, 
               memory_size=3, 
               lr = StepSize(0.3, decay='sqrt', decay_constant=0.01), 
               eps=StepSize(0.7, decay='sqrt', decay_constant=0.001),
#               recovery_procedure=BacktrackingRecovery()
#               convergence_crit=DeltaRelativeError(1e-4, 2)
               )
#agent.plot_pi("./init_pi.png")
result = agent.train(horizon=T, max_episodes=10000,  verbose=True, log_file="./training.log")
#agent.plot_pi("./after_training_pi.png")

