import numpy as np



from pynav import QAgent, BacktrackingAgent
from pynav.environment import Environment
from pynav.feature_maps import FeatureMap
from pynav.recovery_procedures import BacktrackingRecovery, QRecovery

from pynav.utils import StepSize, DeltaRelativeError

import matplotlib.pyplot as plt


data = np.load("nose_data_35_153.npy")
data[data < 3e-6] = 0.0

def s_thr(memory):
    return 3e-6

source_position = np.array([17, 2])
source_radius = (20, 20)
velocity = 20
fmap = FeatureMap(interm_disc=np.array([0.33, 0.66]))
sampling_regions = [
    np.array([[10,25], [40, 100]]),# data.shape[2]]])
#    np.array([[10,40], [60, 80]])# data.shape[2]]])
]
T = 10000
gamma = 0.9

env = Environment(data = data, domain=None, sampling_regions=sampling_regions,  
                  source_position=source_position, source_radius=source_radius, 
                  velocity=velocity, boundary_conditions=True, random_time=True)
agent = BacktrackingAgent(env=env, 
               gamma=gamma, 
               fmap=fmap, 
               s_thr=s_thr, 
               memory_size=5, 
               lr = StepSize(0.1, decay='sqrt', decay_constant=0.01), 
               eps=StepSize(0.50, decay='sqrt', decay_constant=0.01),
               recovery_procedure=QRecovery(time_discretization=np.array([1, 2, 3, 4, 5, 6, 7]), gamma=gamma)
#               convergence_crit=DeltaRelativeError(1e-4, 2)
               )

result = agent.train(horizon=T, max_episodes=5000,  verbose=True, log_file="./training.log")
agent.save("./q_function")

with open("./training.log" , 'r') as f:
    lines = f.readlines()
    rewards = []
    for line in lines:
        rewards.append(float(line.split(",")[1]))
        
        
        
        
rewards = np.array(rewards)


indices = np.array(range(len(rewards)))
neg_mask = rewards < 0.0

#neg_data= np.full(rewards[neg_mask].shape, -1.0)#copy()
#neg_data= -1.0

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(indices[:5000], rewards[:5000] ,'-')

#ax.set_yscale("log")
#ax.set_xscale("log")

#ax.set_xticks(indices)

fig.savefig("./crew_smt.png", bbox_inches='tight')