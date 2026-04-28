import numpy as np
from scipy.stats import uniform

nints = 10
rangespsi0 = np.linspace(-3.5,-1.3,nints)
rangeslogm0 = np.linspace(10.5, 12.7, nints)
rangesmu = np.linspace(7.5,9.7, nints)
rangeseps = np.linspace(0, 1, nints)
rangestau = np.linspace(0.1, 11.0, nints)
rangesgamma = np.linspace(-1.5,0, nints)

#Construct a dictionary with the parameters that I want to vary along with their range of variation
#valid parameter names are:
psi0 = ('psi0',rangespsi0)
psi0name = '$\\psi_0$'
m0 = ('m0',rangeslogm0)
m0name = '$\\log{m_0}$'
mass_norm = ('mass_norm',rangesmu)
mass_normname = '$\\mu$'
scatter = ('scatter', rangeseps)
scattername = '$\\espilon$'
tau = ('tau', rangestau)
tauname = '$\\tau$'
gamma_inner = ('gamma_inner',rangesgamma)
gamma_innername = '$\\gamma_{\\text{inner}}$'
def Parameter_Uniform_Emcee(u,parmax,parmin):
    """Samples a set of parameters within the range defined, from the distribution U(0,1) (For emcee)"""
    if len(parmax) is None:
        pars = u*parmax + (1-u)*parmin
    else:
        pars = u*np.array(parmax)[:,np.newaxis] + (1-u)*np.array(parmin)[:,np.newaxis]
    return pars

def Parameter_Uniform_Dynesty(u,parmax,parmin):
    """Samples a set of parameters within the range defined, from the distribution U(0,1) (For dynesty)"""
    if len(parmax) is None:
        pars = u*parmax + (1-u)*parmin
    else:
        pars = u*np.array(parmax) + (1-u)*np.array(parmin)
    return pars


    
    

