import numpy as np


from pynav.agents import GDAgent, MemGDAgent


data = np.load("/storage/rando/data/nose_data.npy")
source_location = np.array([142, 20])
source_size = 20
gd_agent = MemGDAgent(data=data, source_location=source_location, use_sign = True, source_size=source_size, mem_size=1)


initial_position = np.array([112, 100])
gd_agent.optimize(initial_position, init_t = 0, gamma = 10, horizon=1000, verbose=True)