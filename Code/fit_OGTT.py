import library as lb
import numpy as np

g = np.loadtxt("glucose_control.txt", delimiter=",")
t_data=[0,20,40,60,80,100,120]
glucose_data=g[4, :]


lb.fit_OGTT_ga(t_data, glucose_data)

