import numpy as np

#data = np.load("/storage/rando/data/nose_data.npy")
#valid_init = []

n_thr = 3e-6
#data[data <= n_thr] = 0.0
# with open("/storage/rando/data/valid_init_nose_3e-06", 'w') as f:
#     for k in range(data.shape[0]): # iters on time
#         for i in range(data.shape[1]):
#             for j in range(data.shape[2]):
#                 if data[k, i, j] > n_thr:
#                     print("[{},{},{}] data = {}".format(k, i, j, data[k, i, j]))
#                     f.write("{},{},{}\n".format(k, i, j))
#                     f.flush()

with open("/storage/rando/data/valid_init_nose_3e-06", 'r') as f:
    lines = f.readlines()

domain = []
for line in lines:
    spl = line.split(",")
    domain.append(np.array([float(x) for x in spl]))

domain = np.array(domain).reshape(-1, 3)
np.save("/storage/rando/data/domain_nose_3e-06", domain)