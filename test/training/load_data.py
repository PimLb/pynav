import numpy as np

import matplotlib.pyplot as plt


data = np.load("./nose_data_35_153.npy")

data[data <= 5e-6] = 1e-20

mean_data= np.mean(data, axis=0)

max_location = np.unravel_index(np.argmax(mean_data, axis=None), mean_data.shape)

print(max_location, data[0])

fig, ax = plt.subplots()

ax.imshow(np.log10(data[0, :, :]), origin='lower')
ax.plot(2, 17, 'o', c='red')
fig.savefig("data.pdf", bbox_inches='tight')