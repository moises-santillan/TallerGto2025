import library as lb
import numpy as np

i=4
g = np.loadtxt("glucose_control.txt")
t_data=[0,20,40,60,80,100,120]
glucose_data=g[i, :]


lb.fit_OGTT_ga(t_data, glucose_data)

