import numpy as np 
import math 

a = np.array([1, 1, 1, 2])

mean = np.mean(a)
std = np.var(a)

print((a - mean)/std)