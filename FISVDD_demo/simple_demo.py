from fisvdd import fisvdd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
import pandas as pd


data = pd.read_csv("sample_input.csv", header=0)
data = np.array(data)


s = 0.8 # initialization with a random number between 0~1
fd = fisvdd(data, s)
fd.find_sv()
fd._print_res()

#################
#  plot result  #
#################
plt.scatter(data[:, 0], data[:, 1], s=18)
plt.title('Original Data')
plt.savefig('original_data', dpi=300)
plt.show()
plt.scatter(fd.sv[:, 0], fd.sv[:, 1], s=18)
plt.scatter(fd.sv[:, 0], fd.sv[:, 1], marker='*', s=18)
plt.title('Support Vectors')
plt.savefig('support_vectors', dpi=300)
plt.show()
plt.scatter(data[:, 0], data[:, 1], s=18)
plt.scatter(fd.sv[:, 0], fd.sv[:, 1], marker='*', s=18)
plt.title('Original Data with Support Vectors')
plt.savefig('final_result', dpi=300)
plt.show()
plt.plot(fd.obj_val)
plt.title('Objective Function Value')
plt.savefig('obv', dpi=300)
plt.show()
