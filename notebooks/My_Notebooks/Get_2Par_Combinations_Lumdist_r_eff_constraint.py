import numpy as np
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
import h5py

from utils import TwoDimParVar

nints = 10
rangespsi0 = np.linspace(-3.5,-1.3,nints)
rangeslogm0 = np.linspace(10.5, 12.7, nints)
rangesmu = np.linspace(7.5,9.7, nints)
rangeseps = np.linspace(0, 1, nints)
rangestau = np.linspace(0.1, 11.0, nints)
rangesgamma = np.linspace(-1.5,0, nints)
psi0 = ('psi0',rangespsi0)
m0 = ('m0',rangeslogm0)
mass_norm = ('mass_norm',rangesmu)
scatter = ('scatter', rangeseps)
tau = ('tau', rangestau)
gamma_inner = ('gamma_inner',rangesgamma)

ranges = dict([psi0, m0, mass_norm, scatter, tau, gamma_inner])

#Create a tuple with all possible parameter combinations
subranges = []
for i,v in enumerate(ranges):
    for j,w in enumerate(ranges):
        if i > j:
            subranges.append(dict([(w,list(ranges.values())[j]),(v,list(ranges.values())[i])]))
#For each of these combinations, initialize an object
t = [TwoDimParVar(i,bound_type='Lumdist',model='r_eff', load = True) for i in subranges] 
#Setting load to False will replace all files in the 'Data' folder with new files when the object is called.
f = [i.getFig() for i in t]
#getFig creates figures for a pair of parameters