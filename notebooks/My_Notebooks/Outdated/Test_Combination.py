import numpy as np
import matplotlib.pyplot as plt
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

ranges = dict([psi0, m0])

t = TwoDimParVar(ranges,bound_type='hc',shape=10, load = False)

t.getFig()
print('Done')
