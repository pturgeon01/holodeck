import numpy as np
import matplotlib.pyplot as plt
from holodeck import sams, utils, hardening, host_relations, cosmo
from holodeck.constants import MSOL, SPLC, NWTG, MPC, GYR, YR
from Nanograv_15_year_constraints import interpolate_h_ss_max_Nano_15, interpolate_lumdistmax_Nano_15, get_UL_vs_freq, dl_from_strain
import h5py
from tqdm import tqdm
from scipy import stats
#import emcee

#Nanograv lumdist constraints
#-----------------------------------------------
figdatafile = "Nanograv_data/dist_ul_plot_data.npz"
npzfile = np.load(figdatafile)
xedges = npzfile['xedges']
d_eff_freq = npzfile['d_eff_freq']
d_worst_freq = npzfile['d_worst_freq']
d_best_freq = npzfile['d_best_freq']
dist_UL_freq = npzfile['dist_UL_freq']
#------------------------------------------------
#Nanograv strain constraints
#------------------------------------------------
with h5py.File('Nanograv_data/15yr_quickCW_UL.h5', 'r') as f:
    samples_cold = f['samples_cold'][0,0::1,:]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
log10_fgws = samples_cold[0::1,3]
log10_hs = samples_cold[0::1,4]
#------------------------------------------------

FidPars = dict([('psi0',-2.5),('m0',11.5),('mass_norm',8.3),('scatter',0.45),('tau',5.55),('gamma_inner',-0.75)])
N_to_M = np.loadtxt('Nanograv_data/Num_Reals_to_init_50_e_0095', dtype = int)

class Parameter_Processing():
    """Incorporates all the information necessary to test an arbitrary number of cosmological parameters for a holodeck simulation and return an associated likelihood for these simulations.

    Attributes:
    par_dict (dict): parameter names and their associated value.
    shape (int): shape input into holodeck.
    NUM_FREQS (int): N number of frequency bins input into holodeck.
    NUM_REALS (int): M number of realisations calculated by holodeck.
    NUM_LOUDEST (int): O number of loudest SMBHB pairs returned by holodeck.
    load_strain (bool): load a pre-existing characteristic strain under a specified format.
    varyCM (bool): vary the chirp mass used for luminosity distance calculations.
    full_analysis (bool): returns post-processed characteristic strain.
    bound_type (string): type of bound used (hc, d_lum), obtained from NANOGRAV 15 years.
    model (string): type of luminosity distance bound used when bound_type = d_lum (r_eff, r_allsky).
    """
    def __init__(self, par_dict, shape=30, NUM_FREQS=40, NUM_REALS=50, NUM_LOUDEST=5, load_strain = True,varyCM = True, full_analysis = True, model = 'r_eff', bound_type='hc'):
        self.par_dict = par_dict
        self.shape = shape
        self.NUM_FREQS = NUM_FREQS
        self.NUM_REALS = NUM_REALS
        self.NUM_LOUDEST = NUM_LOUDEST
        self.load_strain = load_strain
        self.varyCM = varyCM
        self.vec_chirp = np.vectorize(utils.chirp_mass_mtmr)
        self.model = model
        self.bound_type = bound_type
        self.full_analysis = full_analysis


    def getFreqs(self):
        """Obtains the Frequency bins and the Frequency bin edges to be used

        Returns:
        fobs (1d numpy array): center of frequency bins
        fobs_edges (1d numpy array): edges of frequency bins
        """
        if not hasattr(self, 'fobs'):
            OBS_DUR = 10*YR
            fobs, fobs_edges = utils.pta_freqs(dur=OBS_DUR, num=self.NUM_FREQS)
            self.fobs = np.array(fobs)
            self.fobs_edges = np.array(fobs_edges)
        return self.fobs, self.fobs_edges

    def Strain(self,return_npz = False):
        """Runs the simulation for the specified parameters to get an associated characteristic strain (background + O loudest single sources) and the associated SMBHB parameters, for each frequency bin (N), .

        Parameters:
        return_npz: Creates a .npz file to store the output

        Returns:
        hc_ss (M x N x O dimensional numpy array): O loudest single source(s) characteristic strain
        hc_bg (M x N dimensional numpy array): background's characteristic strain
        par_ss (4 x M x N x O dimensional numpy array): Holodeck parameters (mtot, q, , z) associated to the O loudest single source(s)
        par_bg (4 x M x N dimensional numpy array): Holodeck parameters (mtot, q, , z) associated to the background's characteristic strain.
        """
        if self.load_strain:
            self.hc_ss = np.load(f'Data/pars_{list(self.par_dict.keys())}.npz', allow_pickle=True)['hc_ss']
            self.hc_bg = np.load(f'Data/pars_{list(self.par_dict.keys())}.npz', allow_pickle=True)['hc_bg']
            self.par_ss = np.load(f'Data/pars_{list(self.par_dict.keys())}.npz', allow_pickle=True)['par_ss']
            self.par_bg = np.load(f'Data/pars_{list(self.par_dict.keys())}.npz', allow_pickle=True)['par_bg']
        else:
            pl = {k: self.par_dict[k] if k in self.par_dict else FidPars[k] for k in FidPars}
            self.getFreqs()
            
            sam = sams.Semi_Analytic_Model(shape=self.shape , gsmf=sams.components.GSMF_Schechter(phi0=pl['psi0'],mchar0_log10=pl['m0']) , mmbulge=host_relations.MMBulge_Standard(mamp_log10=pl['mass_norm'],scatter_dex=pl['scatter']))
            
            self.hc_ss, self.hc_bg, self.par_ss, self.par_bg = sam.gwb(self.fobs_edges,hard = hardening.Fixed_Time_2PL_SAM(sam=sam,time=pl['tau']*GYR,gamma_inner=pl['gamma_inner']), realize=self.NUM_REALS, loudest = self.NUM_LOUDEST, params=True)

        return self.hc_ss, self.hc_bg, self.par_ss, self.par_bg

    def getStrain(self):
        if hasattr(self,'hc_ss'):
            return self.hc_ss, self.hc_bg, self.par_ss, self.par_bg
        else:
            self.Strain()
            return self.hc_ss, self.hc_bg, self.par_ss, self.par_bg
    def getChirpMass(self):
        """Calculates the chirp mass associated to the given loudest single source holodeck parameters from the simulated data.

        Returns:
        Mchirp (N x M x O dimensional numpy array): Chirp masses associated to the O loudest single source(s) across N frequencies and M realisations.
        """
        if not hasattr(self,'Mchirp'):
            self.getStrain()
            mtot = self.par_ss[0]
            mrat = self.par_ss[1]
            self.Mchirp = self.vec_chirp(mtot,mrat)
        return self.Mchirp
    
    def getLumdist(self):
        """Calculates the luminosity distance associated to the given loudest single source holodeck parameters from the simulated data.

        Returns:
        Lumdist (N x M x O dimensional numpy array): Luminosity distances associated to the O loudest single source(s) across N frequencies and M realisations.
        """
        if not hasattr(self,'Lumdist'):
            self.getStrain()
            z = self.par_ss[3]
            self.Lumdist = cosmo.luminosity_distance(z).cgs.value/((1+z)*MPC)
        return self.Lumdist

    def CharStrainBound_Nano_15(self):
        """Calculates the characteristic strain from the GW strain given by Nanograv 15 years.

        Returns:
        h_Nano (N dimensional numpy object): Characteristic strains obtained for each frequency bin
        h_UL_Nano (N dimensional numpy array): Characteristic strain upper bounds for each frequency bin
        bincenters (N dimensional numpy array): Center of each frequency bin
        binedges (N+1 dimensional numpy array): Edges of each frequency bin
        """
        if not hasattr(self,'UL_Nano'):
            self.getFreqs()
            f_min = min(self.fobs)
            f_max = max(self.fobs)
            
            h_Nano, h_UL_Nano, self.bincenters, self.binedges = get_UL_vs_freq(log10_fgws, log10_hs, [f_min,f_max], n_bins=self.NUM_FREQS)
            
            widths = 10**self.binedges[1:] - 10**self.binedges[:-1]
            
            #Convert GW strain to characteristic strain
            self.h_Nano = np.sqrt(np.power(10,self.bincenters)/widths)*h_Nano
            self.h_UL_Nano = np.sqrt(np.power(10,self.bincenters)/widths)*h_UL_Nano

        return self.h_Nano, self.h_UL_Nano, self.bincenters, self.binedges
        
    def LumBound_Nano_15(self):
        """Returns the luminosity distance upper bound from Nanograv 15, in MPC for a range of frequencies and given chirp masses.

        Returns:
        lumdist_UL(N x M x O dimensional numpy array): Luminosity distance (MPC) upper bounds, calculated from the Chirp Mass.
        """        
        if not hasattr(self,'lumdist_UL'):
            self.getFreqs()
            self.getStrain()
            z = self.par_ss[3]
            self.lumdist_UL = interpolate_lumdistmax_Nano_15(self.fobs,z,CM = self.getChirpMass(),model=self.model)
        return self.lumdist_UL

    def Apply_UL_Strain(self, Quick = False, ErrorEst = False):
        """Obtain the realization numbers that contain at least one violation of the characteristic strain lower bound given by Nanograv 15.

        Parameters:
        Quick (bool): Determines whether the quick method for the analysis is done or not
        Returns:
        if Quick:
            realization_number (0-M dimensional numpy array): array with the label of each realization that violates Nanograv's bound
        else:
            hcmax (M dimensional numpy array): Array with the strongest strain per realization relative to the strain upper bound
            frequency_label (M dimensional numpy array): Array with the frequency label of hcmax (Reference value for KDE)
        """
        self.getStrain()
        self.CharStrainBound_Nano_15()
        d = self.hc_ss
        u = self.h_UL_Nano[:,np.newaxis,np.newaxis]
        mask = d > u  
        reals = np.where(mask)[1]                
        self.realization_number = np.unique(reals) #Obtains the realization numbers that satisfy the bound
        M = len(self.realization_number)
        if ErrorEst:
            #I should probably move this segment to another function, to make implementing with R_eff easier.
            if M < 16: #Low number of realizations with a detection
                pass
            elif (len(self.realization_number) > 15) & (len(self.realization_number) < 46): 
                #Adapt total number of realizations to match provided list
                #Eventially if I want to increase the initial number of realizations I can just implement the Like-Optimization function somewhere.
                
                N_i = self.NUM_REALS #Initial number of realizations
                self.NUM_REALS = N_to_M[int(M-16)] - N_i #Amount of realizations to add
                self.hc_ss = np.concatenate((d,self.Strain()[0]),axis = 1) #Add new realizations to the strain
                #Eventually add new parameters/backgrounds from new realizations
                self.NUM_REALS = N_to_M[int(M-16)] #Total number of realizations
                d = self.hc_ss 
                mask = d > u  
                reals = np.where(mask)[1]
                self.realization_number = np.unique(reals) #Re-obtain statistic
            else:
                #Same steps as before, but with an upper threshold of 1000 realizations
                
                N_i = self.NUM_REALS
                self.NUM_REALS = 1000 - N_i
                self.hc_ss = np.concatenate((d,self.Strain()[0]),axis = 1)
                #Eventually add new parameters/backgrounds from new realizations
                self.NUM_REALS = 1000
                d = self.hc_ss 
                mask = d > u  
                reals = np.where(mask)[1]   
                self.realization_number = np.unique(reals)

        if Quick:                        
            return self.realization_number
        else:
            #I want the strain for each realization that maximizes the strain/strain_95 rate and its frequency number which will then be used with the KDE (at its respective frequency) to obtain its pdf
            strongest_signal = np.max(d/u, axis = (0,2))
            loc = np.transpose(np.where(d/u == strongest_signal[np.newaxis,:,np.newaxis]))
            sortedloc = loc[np.argsort(loc[:,1])] #Indices sorted by realisations
            self.frequency_label = sortedloc[:,0] #Frequency index of every strongest signal
            self.hcmax = self.hc_ss[sortedloc[:,0],sortedloc[:,1],sortedloc[:,2]] #Strain value of every strongest signal
            
            return self.hcmax, self.frequency_label

    def Apply_UL_Lumdist(self, Quick = False):
        """Obtain the realization numbers that contain at least one violation of the luminosity distance upper bound given by Nanograv 15.
        
        Parameters:
        Quick (bool): Determines whether the quick method for the analysis is done or not
        Returns:
        realization_number (0-M dimensional numpy array): array with the label of each realization that violates Nanograv's bound
        """
        d = self.getLumdist()
        u = self.LumBound_Nano_15()
        mask = d < u                          
        reals = np.where(mask)[1]                
        self.realization_number = np.unique(reals) #Obtains the realization numbers that satisfy the bound
        return self.realization_number

    def Quick_Likelihood(self):
        """Obtain the likelihood estimate based on the probability of CW non-detection at a 95 % confidence interval of the Nanograv 15 data.
        
        Returns:
        l (float): Likelihood estimate
        """
        if self.bound_type == 'hc':
            self.l = 1 - len(self.Apply_UL_Strain(Quick = True))/self.NUM_REALS
            print(f'Using the characteristic strain bound provided by Nanograv, the likelihood can be estimated as {self.l}') 
            return self.l
        elif self.bound_type == 'Lumdist':
            self.l = 1 - len(self.Apply_UL_Lumdist(Quick = True))/self.NUM_REALS
            print(f'Using the luminosity distance bound provided by Nanograv, the likelihood can be estimated as {self.l}')
            return self.l
        else:
            raise ValueError('Please use a correct bound_type')

    def get_Strain_pdf(self, provide_kde = False):
        """Returns the strain probability density function (pdf) for each frequency bin, realisation number and loudest , which can then be processed in Bayesian_Methods.

        Returns:
        pdf ((len(par_dict) x M x N x O dimensional array): a pdf array which can be processed using other methods 
        provide_kde (bool/M dimensional array): If the shape of the constraints vector does not change across multiple objects, a pre-loaded vector can be provided to avoid having to re-calculate the KDE
        """
        
        if not hasattr(self,'pdf'):
            hc_max, freq_idx = self.Apply_UL_Strain(Quick = False)
            pdf = np.empty(np.shape(hc_max))
            if isinstance(provide_kde, bool) and provide_kde == False:
                self.CharStrainBound_Nano_15()
                kdes = [stats.gaussian_kde(h) for h in self.h_Nano]
            else:
                kdes = [stats.gaussian_kde(h) for h in provide_kde]
            for i,v in enumerate(hc_max):           
                vals = kdes[freq_idx[i]](v) + kdes[freq_idx[i]](-v)
                pdf[i] = vals
            self.pdf = pdf.reshape((1,) * len(self.par_dict) + pdf.shape)
        return self.pdf
            

        
        
    
        
