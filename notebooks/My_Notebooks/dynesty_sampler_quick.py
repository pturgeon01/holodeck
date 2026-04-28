import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import Bayesian_Methods as Bayes
import Parameters as Pars
from Full_Par_Method import Parameter_Processing
from dynesty import NestedSampler

par_name = (Pars.psi0[0], Pars.m0[0], Pars.mass_norm[0], Pars.scatter[0], Pars.tau[0], Pars.gamma_inner[0])
parrange = ((-3.5,10.5,7.5,0,0.1,-1.5),(-1.5,12.5,9.5,1,11,0))
ndim = 6

def Likelihood_Wrapper(parameters, par_names = par_name):
    par_dict = {key: value for key, value in zip(par_names,parameters)}
    likelihood = Parameter_Processing(par_dict, load_strain = False).Quick_Likelihood()
    return np.log10(likelihood)

def Prior_Wrapper(u, parmax = parrange[1], parmin = parrange[0]):
    return Pars.Parameter_Uniform(u,parmax,parmin)

sampler = NestedSampler(Likelihood_Wrapper, Prior_Wrapper, ndim = ndim)
sampler.run_nested()
sampler.save(f'Dynesty_Samples/{par_name}')
