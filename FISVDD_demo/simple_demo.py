from fisvdd import fisvdd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
import pandas as pd


data = pd.read_csv("banana_ol.csv", header=0)
data = np.array(data)

s = 0.8  # initialization with a random number between 0~1
fd = fisvdd(data, s)
fd.find_sv()
fd._print_res()

# plot result
plt.scatter(data[:, 0], data[:, 1], s=30)
plt.scatter(fd.sv[:, 0], fd.sv[:, 1], marker='*')
plt.title('original data with detected boundary')
plt.show()
plt.plot(fd.obj_val)
plt.show()
