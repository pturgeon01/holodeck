import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import Bayesian_Methods as Bayes
import Parameters as Pars
from functools import partial
import sys
from Full_Par_Method import Parameter_Processing
import Full_Par_Method
from multiprocess import Pool
from dynesty import DynamicNestedSampler, sampler, plotting
from getdist import plots, MCSamples

par_names = (Pars.psi0[0], Pars.m0[0], Pars.mass_norm[0], Pars.scatter[0], Pars.tau[0], Pars.gamma_inner[0])
parrange = ((-3.5,10.5,7.5,0,0.1,-1.5),(-1.5,12.5,9.5,1,11,0))
ndim = 6
Data = Parameter_Processing(Full_Par_Method.FidPars, load_strain = False)
Nano_strains = Data.CharStrainBound_Nano_15()[0]

def Likelihood_Wrapper(parameters):
    par_dict = {key: value for key, value in zip(par_names,parameters)}
    t1 = Parameter_Processing(par_dict, load_strain = False)
    t1.get_Strain_pdf(provide_kde = Nano_strains)
    DistObj = Bayes.Distribution(par_dict) #Initialize Distribution Object for the pre-defined parameters
    PDF = DistObj.PDF(DistObj) #Create a pdf object from the given distribution object
    pdf = PDF.get_pdf(t1.pdf, t1.hcmax) #Obtain the pdf distribution as an array
    PDF.dim_names = np.array('Realisations') #Set the name of the dimensions of the PDF to the dimensions of holodeck
    Priors = DistObj.Priors(DistObj)
    Priors.Uniform()
    I = DistObj.MCI(Priors,PDF)  
    I.apply_prior()
    Likelihood = I.Integration('Realisations')
    return np.log(Likelihood.flatten()[0])

def Prior_Wrapper(u):
    return Pars.Parameter_Uniform(u,parrange[1],parrange[0])

prior = partial(Prior_Wrapper, Pars = u)
likelihood = partial(Likelihood_Wrapper, 
if __name__ == '__main__':
    with Pool(processes=8) as POOL:
        sampler = DynamicNestedSampler(Likelihood_Wrapper, Prior_Wrapper, pool = POOL, queue_size = 8, ndim = 1)
        sampler.run_nested(n_effective = 1)
    #sampler.save(f'Dynesty_Samples/{par_names}_Full_Like_dynamic')