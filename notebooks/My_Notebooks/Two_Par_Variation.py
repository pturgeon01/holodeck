import numpy as np
import matplotlib.pyplot as plt
from holodeck import sams, utils, hardening, host_relations, cosmo
from holodeck.constants import MSOL, SPLC, NWTG, MPC, GYR, YR
from Nanograv_15_year_constraints import interpolate_h_ss_max_Nano_15, get_UL_vs_freq, dl_from_strain
import h5py
from tqdm import tqdm
from scipy import stats
import emcee

#Nanograv lumdist constraints
#-----------------------------------------------
figdatafile = "../../data/dist_ul_plot_data.npz"
npzfile = np.load(figdatafile)
xedges = npzfile['xedges']
d_eff_freq = npzfile['d_eff_freq']
d_worst_freq = npzfile['d_worst_freq']
d_best_freq = npzfile['d_best_freq']
dist_UL_freq = npzfile['dist_UL_freq']
#------------------------------------------------
#Nanograv strain constraints
#------------------------------------------------
with h5py.File('../../data/15yr_quickCW_UL.h5', 'r') as f:
    samples_cold = f['samples_cold'][0,0::1,:]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
log10_fgws = samples_cold[0::1,3]
log10_hs = samples_cold[0::1,4]
#------------------------------------------------

FidPars = dict([('psi0',-2.5),('m0',11.5),('mass_norm',8.3),('scatter',0.45),('tau',5.55),('gamma_inner',-0.75)])

class TwoDimParVar():
    """Incorporates all the information necessary to test two cosmological parameters for a holodeck simulation and return an associated likelihood for these simulations."""
    def __init__(self, par_ranges, shape=30, NUM_FREQS=40, NUM_REALS=100, NUM_LOUDEST=5, load = True,varyCM = True, full_analysis = True, model = 'r_eff', bound_type='hc'):
        self.par_ranges = par_ranges 
        self.shape = shape
        self.NUM_FREQS = NUM_FREQS
        self.NUM_REALS = NUM_REALS
        self.NUM_LOUDEST = NUM_LOUDEST
        self.load = load
        self.varyCM = varyCM
        self.vec_chirp = np.vectorize(utils.chirp_mass_mtmr)
        self.model = model
        self.bound_type = bound_type
        self.full_analysis = full_analysis
        
        names = [[],[]]
        if 'psi0' in list(par_ranges.keys()):
            names[list(par_ranges).index('psi0')] = r'$\psi_0$'
        
        if 'm0' in list(par_ranges.keys()):
            names[list(par_ranges).index('m0')] = r'$\log{m_0}$'
        
        if 'mass_norm' in list(par_ranges.keys()):
            names[list(par_ranges).index('mass_norm')] = r'$\mu$'
        
        if 'scatter' in list(par_ranges.keys()):
            names[list(par_ranges).index('scatter')] = 'ϵ'
        
        if 'tau' in list(par_ranges.keys()):
            names[list(par_ranges).index('tau')] = r'$\tau$'
        
        if 'gamma_inner' in list(par_ranges.keys()):
            names[list(par_ranges).index('gamma_inner')] = r'$\gamma_{\text{inner}}$'
        self.names = names
        
    def getFreqs(self):
        """Obtains the Frequency bins and the Frequency bin edges to be used"""
        if not hasattr(self, 'fobs'):
            OBS_DUR = 10*YR
            fobs, fobs_edges = utils.pta_freqs(dur=OBS_DUR, num=self.NUM_FREQS)
            self.fobs = fobs
            self.fobs_edges = fobs_edges
        return self.fobs, self.fobs_edges


        
    def getStrain(self,returnnpz = True):
        """Runs the simulation for each specified parameter pair across a grid to get the strain (background + loudest single source) and the associated SMBHB parameters."""
        if hasattr(self,'hc_ss'):
            return self.hc_ss, self.hc_bg, self.par_ss, self.par_bg
        elif self.load:
                hc_ss = np.load(f'Data/pars_{list(self.par_ranges.keys())}.npz', allow_pickle=True)['hc_ss']
                hc_bg = np.load(f'Data/pars_{list(self.par_ranges.keys())}.npz', allow_pickle=True)['hc_bg']
                par_ss = np.load(f'Data/pars_{list(self.par_ranges.keys())}.npz', allow_pickle=True)['par_ss']
                par_bg = np.load(f'Data/pars_{list(self.par_ranges.keys())}.npz', allow_pickle=True)['par_bg']
        else:
            v0, v1 = list(self.par_ranges.values())
            k0, k1 = list(self.par_ranges.keys())
            l0, l1 = len(v0), len(v1)
            if k0 == k1:
                raise ValueError("Please vary two distinct parameters")  
            elif not isinstance(self.par_ranges,dict) or len(self.par_ranges) != 2:
                raise ValueError("\'ranges\' must be of type dictionary and length 2.")
            elif k0 not in list(FidPars.keys()) or k1 not in list(FidPars.keys()):
                raise ValueError(f"Both varied parameters must be one of:{list(FidPars.keys())}")
            else:
                sam = np.zeros((l0,l1), dtype='object')
                hc_ss = np.zeros((l0,l1), dtype='object')
                hc_bg = np.zeros((l0,l1), dtype='object')
                par_ss = np.zeros((l0,l1), dtype='object')
                par_bg = np.zeros((l0,l1), dtype='object')
                log0 = np.zeros(len(FidPars))
                log1 = np.zeros(len(FidPars))
                FidParsl = list(FidPars.values())
                fobs_edges = self.getFreqs()[1]
                
                #Set each logging array to be 1 for the varied parameter
                log0[list(FidPars).index(k0)] = 1 
                log1[list(FidPars).index(k1)] = 1
                        
                for i,v in enumerate(v0): #Iterate over the range of the first variable
                    print(f"GW single source strain calculation is {i/l0*100}% done.")
                    for j,w in enumerate(v1): #Iterate over the range of the second variable
                        pn = []
                        for k,x in enumerate(FidParsl):
                            pn.append(x*(v/x)**(log0[k])*(w/x)**(log1[k])) #Gives a parameter that is not set to the fiducial value only if set in the ranges array
                            
                        sam[i,j] = sams.Semi_Analytic_Model(shape=self.shape , gsmf=sams.components.GSMF_Schechter(phi0=pn[0],mchar0_log10=pn[1]) , mmbulge=host_relations.MMBulge_Standard(mamp_log10=pn[2],scatter_dex=pn[3]))
                        
                        hc_ss[i,j], hc_bg[i,j], par_ss[i,j], par_bg[i,j] = sam[i,j].gwb(fobs_edges,hard = hardening.Fixed_Time_2PL_SAM(sam=sam[i,j],time=pn[4]*GYR,gamma_inner=pn[5]), realize=self.NUM_REALS, loudest = self.NUM_LOUDEST, params=True)
        self.hc_ss = hc_ss
        self.hc_bg = hc_bg
        self.par_ss = par_ss
        self.par_bg = par_bg
        if returnnpz:
            self.getnpz()
                
            
        return self.hc_ss, self.hc_bg, self.par_ss, self.par_bg

    def getChirpMass(self):
        if not hasattr(self,'Mchirp'):
            mtot = self.flattenPars()[2][:,:,0]
            mrat = self.flattenPars()[2][:,:,1]
            self.Mchirp = self.vec_chirp(mtot,mrat)
        return self.Mchirp
    
    def getLumdist(self):
        if not hasattr(self,'Lumdist_parvar'):
            z = self.flattenPars()[2][:,:,3]
            self.Lumdist_parvar = cosmo.luminosity_distance(z).cgs.value/((1+z)*MPC)
        return self.Lumdist_parvar
    
    def flattenPars(self):
        """Flattens the strain and parameter tuple, useful for vectorization."""
        if not hasattr(self, 'hc_ss'):
            hc_ss, hc_bg, par_ss, par_bg = self.getStrain(returnnpz=False)
        hc_ss = np.array(self.hc_ss.tolist())
        hc_bg = np.array(self.hc_bg.tolist())
        par_ss = np.array(self.par_ss.tolist())
        par_bg = np.array(self.par_bg.tolist())
        return hc_ss, hc_bg, par_ss, par_bg
        
    def getNano_15_strain(self):
        if not hasattr(self,'UL_freq'):
            f_min = 1e-9
            f_max = 3e-7
            Dist_freq, UL_freq, bincenters, binedges = get_UL_vs_freq(log10_fgws, log10_hs, [f_min,f_max], n_bins=self.NUM_FREQS)
            
            widths = 10**binedges[1:] - 10**binedges[:-1]
            
            #Convert GW strain to characteristic strain
            Dist_freq = np.sqrt(np.power(10,bincenters)/widths)*Dist_freq
            UL_freq = np.sqrt(np.power(10,bincenters)/widths)*UL_freq

        return Dist_freq, UL_freq, bincenters, binedges

    def get_Strain_pdf(self,returnnpz = False,loadpdf = True):
        """Returns the strain pdf that depends on a given frequency bin"""
        if not hasattr(self,'pdf'):
            if loadpdf:
                pdf = np.load(f'hc_PDF/pars_{list(self.par_ranges.keys())}_pdf.npz', allow_pickle=True)['pdf']
                self.pdf = np.array(pdf, dtype='float')
            else:
                Dist_freq, UL_freq, bincenters, xedges2 = self.getNano_15_strain()
                hc_ssf = np.array(self.flattenPars()[0])
                fobs, f_edges = self.getFreqs()
                pdf = np.empty(hc_ssf.shape, dtype=object)
                kdes = [stats.gaussian_kde(h) for h in Dist_freq]
                for i, kde in enumerate(tqdm(kdes)):
                    block = hc_ssf[:, :, i, :, :]
                    pts = block.ravel()           
                    vals = kde(pts) + kde(-pts)
                    pdf[:, :, i, :, :] = vals.reshape(block.shape)
                self.pdf = np.array(pdf, dtype='float')
            if returnnpz:
                self.getnpz(descriptor = 'pdf')
        return self.pdf

    def StrainBound_Nano_15(self):
        """Returns the single source strain 90% CI as the lower bound for a range of frequencies"""
        Dist_freq, UL_freq, bincenters, binedges = self.getNano_15_strain()
        fobs, f_edges = self.getFreqs()
        Upper_boundsh = interpolate_h_ss_max_Nano_15(self.getFreqs()[0],binedges,UL_freq)
        Upper_boundshc = np.array([np.sqrt(v/(f_edges[i+1]-f_edges[i]))*Upper_boundsh[i] for i,v in enumerate(fobs)])
        return Upper_boundshc
        
    def LumBound_Nano_15(self, CM = float(1e9*MSOL)):
        '''Returns the luminosity distance upper bound in MPC for a range of frequencies and given chirp masses.'''
        f = self.getFreqs()[0]
        z = self.flattenPars()[2][:,:,3]
        eff_lumdist = np.zeros(np.shape(f)[0])
        if self.model == 'strain':
            eff_lumdist =  dl_from_strain(f,self.getStrain[0],self.getStrain[2],mchirp=self.getChirpMass())
            return eff_lumdist
        for i,v in enumerate(np.log10(f)):
            mindiff = np.inf #Start with arbitrarily large interval
            for j,w in enumerate(xedges[:-1]): #Iterate over frequency
                if np.abs(v - (w + xedges[j+1])/2 ) < mindiff: #Fixes offset of frequency bin
                    mindiff = np.abs(v - (w + xedges[j+1])/2)
                    mindifflab = j
            if self.model == 'all_sky':
                eff_lumdist[i] = dist_UL_freq[mindifflab]
            elif self.model == 'r_eff':
                eff_lumdist[i] = d_eff_freq[mindifflab]
            elif self.model == 'worst':
                eff_lumdist[i] = d_worst_freq[mindifflab] 
            elif self.model == 'best':
                eff_lumdist[i] = d_best_freq[mindifflab] 
            else: 
                raise ValueError('Please use a valid upper bound model.')
        if isinstance(CM, float) and CM == 1e9*MSOL:
            return eff_lumdist
            
        elif isinstance(CM, float) and CM != 1e9*MSOL:
            eff_lumdist = eff_lumdist*(CM*(1+z))/(1e9*MSOL)**(5/3)
            return eff_lumdist
            
        elif isinstance(CM, np.ndarray):
            lumdist_upper_limits = eff_lumdist[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]*(CM*(1+z)/(1e9*MSOL))**(5/3)
            return lumdist_upper_limits
            
        else:
            raise ValueError('Input parameters not of the correct form.')

            
    def Detectable_Sources_Lumdist(self,d,u,dpars=False):
        #Optimize
        '''Takes as input a frequency array (1D), the upper bound array (3D) and the luminosity distance array (3D). Returns the number of the realizations that have at least one detection. Optinally, the luminosity distance of detectable sources (3D), their corresponding frequencies (3D), their corresponding distance upper limit their strain can also be obtained; along with a 1d array of a detection's frequency. '''

        FilteredLumdist = np.zeros(np.shape(d))
        FilteredFreqs = np.zeros(np.shape(d))
        corresponding_upper_limit = np.zeros(np.shape(d))
        realization_number = []
        frequency_of_SMBHB = []
        
        if not self.varyCM:
            for i in np.arange(np.shape(d)[0]): #Iterates over each frequency 
                for j in np.arange(np.shape(d)[1]): #Iterates over each realization 
                    for k in np.arange(np.shape(d)[2]): #Iterates over each loudest SMBHB
                        if d[i,j,k] < u[i]:
                            realization_number.append(j)
                            if not dpars:
                                frequency_of_SMBHB.append(self.getFreqs()[0][i])
                                FilteredLumdist[i,j,k] = d[i,j,k]
                                FilteredFreqs[i,j,k] = self.getFreqs()[0][i]
                                corresponding_upper_limit[i,j,k] = u[i,j,k]
        
        else:
            for i in np.arange(np.shape(d)[0]): #Iterates over each frequency 
                for j in np.arange(np.shape(d)[1]): #Iterates over each realization 
                    for k in np.arange(np.shape(d)[2]): #Iterates over each loudest SMBHB
                        if d[i,j,k] < u[i,j,k]:
                            realization_number.append(j)
                            if not dpars:
                                frequency_of_SMBHB.append(self.getFreqs()[0][i])
                                FilteredLumdist[i,j,k] = d[i,j,k]
                                FilteredFreqs[i,j,k] = self.getFreqs()[0][i]
                                corresponding_upper_limit[i,j,k] = u[i,j,k]
        
            return FilteredLumdist, FilteredFreqs, corresponding_upper_limit, np.array(realization_number), np.array(frequency_of_SMBHB)
    
    
    def Detectable_Sources_Strain(self,hc_ss,dpars=False):
        hc_ss_b = self.StrainBound_Nano_15() 
        Filteredhc_ss = np.zeros(np.shape(hc_ss))
        FilteredFreqs = np.zeros(np.shape(hc_ss))
        corresponding_upper_limit = np.zeros(np.shape(hc_ss))
        realization_number = []
        frequency_of_SMBHB = []
        if not self.full_analysis:
            for i in np.arange(np.shape(hc_ss)[0]): #Iterates over each frequency 
                for j in np.arange(np.shape(hc_ss)[1]): #Iterates over each realization 
                    for k in np.arange(np.shape(hc_ss)[2]): #Iterates over each loudest SMBHB
                        if hc_ss[i,j,k] > hc_ss_b[i]:
                            realization_number.append(j)
                            if not dpars:
                                frequency_of_SMBHB.append(self.getFreqs()[0][i])
                                Filteredhc_ss[i,j,k] = hc_ss[i,j,k]
                                FilteredFreqs[i,j,k] = self.getFreqs()[0][i]
                                corresponding_upper_limit[i,j,k] = hc_ss_b[i]
        else:
            raise ValueError('Standard bounds can not be applied to determine detectability of strain when incorporating the full analysis.')
            
        return Filteredhc_ss, FilteredFreqs, corresponding_upper_limit, np.array(realization_number), np.array(frequency_of_SMBHB)
        
        
    def getAcceptedVals(self,dpars = False):
        """Applies the above function to a grid of 2 cosmological parameters."""
        hc_ss, hc_bg, par_ss, par_bg = self.getStrain(returnnpz=False)
        hc_ssf, hc_bgf, par_ssf, par_bgf = self.flattenPars()
        
        Filteredobs_var = np.zeros(np.shape(hc_ssf))
        FilteredFreqs_var = np.zeros(np.shape(hc_ssf))
        corresponding_upper_limit_var = np.zeros(np.shape(hc_ssf))
        realization_number_var = np.zeros((np.shape(hc_ss)[0],np.shape(hc_ss)[1]),dtype='object')
        frequency_of_SMBHB_var = np.zeros((np.shape(hc_ss)[0],np.shape(hc_ss)[1]),dtype='object')
        
        if self.bound_type == 'Lumdist':
            Lumdist_parvar = self.getLumdist()
            if self.varyCM == True:
                Upper_boundsL = self.LumBound_Nano_15(CM = self.getChirpMass())
                for i in range(np.shape(hc_ss)[0]):
                    for j in range(np.shape(hc_ss)[1]):
                        Filteredobs_var[i,j], FilteredFreqs_var[i,j], corresponding_upper_limit_var[i,j], realization_number_var[i,j], frequency_of_SMBHB_var[i,j] = self.Detectable_Sources_Lumdist(Lumdist_parvar[i,j],Upper_boundsL[i,j], dpars = dpars)
            else:
                Upper_boundsL = self.LumBound_Nano_15()
                for i in range(np.shape(hc_ss)[0]):
                    for j in range(np.shape(hc_ss)[1]):
                        Filteredobs_var[i,j], FilteredFreqs_var[i,j], corresponding_upper_limit_var[i,j], realization_number_var[i,j], frequency_of_SMBHB_var[i,j] = self.Detectable_Sources_Lumdist(Lumdist_parvar[i,j],Upper_boundsL, dpars = dpars)
            
        elif self.bound_type == 'hc':
            hc_parvar = self.getStrain()[0]
            for i in range(np.shape(hc_ss)[0]):
                    for j in range(np.shape(hc_ss)[1]):
                        Filteredobs_var[i,j], FilteredFreqs_var[i,j], corresponding_upper_limit_var[i,j], realization_number_var[i,j], frequency_of_SMBHB_var[i,j] = self.Detectable_Sources_Strain(hc_parvar[i,j], dpars = dpars)
                        
        return Filteredobs_var, corresponding_upper_limit_var, realization_number_var, frequency_of_SMBHB_var

        
    def CalculateLikelihood(self, Succ_Real_N):
        """Takes as input the object containing the realization numbers with at least one detection for each parameter pair (N,M). \n
        Returns a 2D array containing the Likelihood values """
        c1_var = np.zeros(np.shape(Succ_Real_N),dtype='object')
        for i,v in enumerate(Succ_Real_N):
            for j,w in enumerate(v):
                c1_var[i,j] = np.histogram(w,bins=self.NUM_REALS)[0]
        success1_var = np.zeros(np.shape(Succ_Real_N))
        total1_var = np.zeros(np.shape(Succ_Real_N))
        for i,v in enumerate(c1_var):
            for j,w in enumerate(v):
                total1_var[i,j] = len(w)
                for k in w:
                    if k != 0:
                        success1_var[i,j] += 1
        success1_var = np.array(success1_var.tolist())
        total1_var = np.array(total1_var.tolist())
        self.Likelihoods = 1 - success1_var/total1_var
        return self.Likelihoods


    def getnpz(self,descriptor = None):
        """Writes the single source characteristic strain, background characteristic strain and their associated parameters ont a .npz file."""
        self.getStrain(returnnpz=False)
        if descriptor == 'pdf':
            np.savez(f'hc_PDF/pars_{list(self.par_ranges.keys())}_{descriptor}', pdf = self.pdf, hc_ss = self.hc_ss)
        else:
            np.savez(f'Data/pars_{list(self.par_ranges.keys())}', hc_ss = self.hc_ss, hc_bg = self.hc_bg, par_ss = self.par_ss, par_bg = self.par_bg)

    def getFig(self):
        """Obtains a 2D heatmap for each associated parameter"""
        figs, ax = plt.subplots(layout='constrained')
        figsize=(16,4)
        im = ax.imshow(self.CalculateLikelihood(self.getAcceptedVals()[2]), cmap='hot', interpolation='nearest',extent = (list(self.par_ranges.values())[1][0],list(self.par_ranges.values())[1][-1],list(self.par_ranges.values())[0][-1],list(self.par_ranges.values())[0][0]), aspect = 'auto')
        ax.set_xlabel(self.names[1],fontsize=16)
        ax.set_ylabel(self.names[0],fontsize=16)
        figs.colorbar(im,label='Likelihood')
        if self.bound_type == 'hc':
            plt.savefig(f'Figures2D_{self.bound_type}/Likelihood_Heatmap_pars_{list(self.par_ranges.keys())}_boundtype_{self.bound_type}')
        else:
            plt.savefig(f'Figures2D_{self.bound_type}_{self.model}/Likelihood_Heatmap_pars_{list(self.par_ranges.keys())}_boundtype_{self.bound_type}_model_{self.model}')
        

        
