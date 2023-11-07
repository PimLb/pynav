import numpy as np

import h5py 


fl = h5py.File("./nose_data_35_153.mat")

data = np.array(fl['nose'])
data = np.delete(data, 1732, axis=0)
data = np.delete(data, 2460, axis=0) 
np.save("nose_data_35_153", data, allow_pickle=True)
valid_init = []

n_thr = 3e-6
with open("init_nose_3e-06", 'w') as f:
    for k in range(data.shape[0]): # iters on time
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if data[k, i, j] > n_thr:
                    print("[{},{},{}] data = {}".format(k, i, j, data[k, i, j]))
                    f.write("{},{},{}\n".format(k, i, j))
                    f.flush()
