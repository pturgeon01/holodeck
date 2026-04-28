#Initialize the npz file for the luminosity distance limits
import numpy as np
import matplotlib.pyplot as plt
from holodeck import sams, utils, hardening, host_relations
from holodeck.constants import MSOL, SPLC, NWTG, MPC, GYR
import h5py
import emcee

figdatafile = "../Nanograv_data/dist_ul_plot_data.npz"
npzfile = np.load(figdatafile)
xedges = npzfile['xedges']
d_eff_freq = npzfile['d_eff_freq']
d_worst_freq = npzfile['d_worst_freq']
d_best_freq = npzfile['d_best_freq']
dist_UL_freq = npzfile['dist_UL_freq']

def get_UL_vs_freq(log10_fgws, log10_hs, f_bounds, n_bins=37):
    f_min = f_bounds[0]
    f_max = f_bounds[1]
    
    f_bins = np.logspace(np.log10(f_min), np.log10(f_max), n_bins+1)

    f_bincenters = []
    for i in range(f_bins.size-1):
        f_bincenters.append((f_bins[i+1]+f_bins[i])/2)
    f_bincenters = np.array(f_bincenters)

    log10_h_bins = np.linspace(-18,-11,100)

    h, xedges2, yedges = np.histogram2d(log10_fgws, log10_hs, bins=[np.log10(f_bins), log10_h_bins])

    #make bin centers
    bincenters = []
    for i in range(xedges2.size-1):
        bincenters.append((xedges2[i+1]+xedges2[i])/2)
    bincenters = np.array(bincenters)

    freq_idx = np.digitize(log10_fgws, xedges2)

    Dist_freq = np.zeros(bincenters.size,dtype='object')
    UL_freq = np.zeros(bincenters.size)
    for i in range(bincenters.size):
        hs = 10**log10_hs[np.where(freq_idx==i+1)]
        if hs.size==0:
            Dist_freq[i] = 0.0
            continue

        #normal UL
        Dist_freq[i] = hs
        UL_freq[i] = np.percentile(hs, 95)

    return Dist_freq, UL_freq, bincenters, xedges2

def dl_from_strain(f,h,z,mchirp=10**9):
    frest = utils.frst_from_fobs(f[:,np.newaxis,np.newaxis], z) #Convert f to rest frame
    return _GW_SRC_CONST * mchirp * np.power(mchirp*frest, 2/3) / (h*MPC) * (1 + z)

def interpolate_lumdistmax_Nano_15(f,z,CM = float(1e9*MSOL),model='all_sky'):
    '''Returns the luminosity distance upper bound in MPC for a range of frequencies and given chirp masses.'''
    eff_lumdist = np.zeros(np.shape(f)[0])
    for i,v in enumerate(np.log10(f)):
        mindiff = np.inf #Start with arbitrarily large interval
        for j,w in enumerate(xedges[:-1]): #Iterate over frequency
            if np.abs(v - (w + xedges[j+1])/2 ) < mindiff: #Fixes offset of frequency bin
                mindiff = np.abs(v - (w + xedges[j+1])/2)
                mindifflab = j
        if model == 'all_sky':
            eff_lumdist[i] = dist_UL_freq[mindifflab]
        elif model == 'r_eff':
            eff_lumdist[i] = d_eff_freq[mindifflab]
        elif self.model == 'worst':
            eff_lumdist[i] = d_worst_freq[mindifflab] 
        elif self.model == 'best':
            eff_lumdist[i] = d_best_freq[mindifflab] 
    
    #Set Chirp mass to observer frame chirp mass
    if isinstance(CM, float) and CM == 1e9*MSOL:
        return eff_lumdist
    elif isinstance(CM, float) and CM != 1e9*MSOL:
        eff_lumdist = eff_lumdist*(CM*(1+z))/(1e9*MSOL)**(5/3)
        return eff_lumdist
    elif isinstance(CM, np.ndarray):
        lumdist_upper_limits = np.zeros(np.shape(CM))
        lumdist_upper_limits = eff_lumdist[:,np.newaxis,np.newaxis]*(CM*(1+z)/(1e9*MSOL))**(5/3)
        return lumdist_upper_limits
    else:
        raise ValueError('Input parameters not of the correct form.')

def get_ss_GW_strain(f,f_edges,hc_ss):
    '''Returns the single source GW strain from the characteristic strain (3D hc_ss)'''
    h_ss = np.zeros(np.shape(hc_ss))
    for i in np.arange(np.shape(h_ss)[0]):
        h_ss[i,:,:] = hc_ss[i,:,:]*np.sqrt((f_edges[i+1]-f_edges[i])/f[i]) 
    return h_ss

def get_bg_GW_strain(f,f_edges,hc_bg):
    '''Returns the background GW strain from the characteristic strain'''
    h_bg = np.zeros(np.shape(hc_bg))
    for i in np.arange(np.shape(h_bg)[0]):
        h_bg[i,:] = hc_bg[i,:]*np.sqrt((f_edges[i+1]-f_edges[i])/f[i]) 
    return h_bg

def get_ss_Char_strain(f,f_edges,h_ss):
    '''Returns the characteristic strain from the single source GW strain (1D h_ss)'''
    hc_ss = np.zeros(np.shape(h_ss))
    for i in np.arange(np.shape(h_ss)[0]):
        hc_ss[i] = h_ss[i]*np.sqrt(f[i]/(f_edges[i+1]-f_edges[i])) 
    return hc_ss


def interpolate_h_ss_max_Nano_15(f,h_ss,f_edges=xedges):
    '''Returns the strain upper bound in MPC for a range of frequencies and given chirp masses.'''
    eff_h_ss = np.zeros(np.shape(f)[0])
    #hc_ss = get_ss_Char_strain(f,f_edges,h_ss)
    for i,v in enumerate(np.log10(f)):
        mindiff = np.inf #Start with arbitrarily large interval
        for j,w in enumerate(f_edges[:-1]): #Iterate over frequency
            if np.abs(v - (w + f_edges[j+1])/2 ) < mindiff: #Fixes offset of frequency bin
                mindiff = np.abs(v - (w + f_edges[j+1])/2)
                mindifflab = j
        eff_h_ss[i] = h_ss[mindifflab]
    return eff_h_ss


def Detectable_Sources_Lumdist(f,d,u,h=None):
    '''Takes as input a frequency array (1D), the upper bound array (3D) and the luminosity distance array (3D). Returns the luminosity distance of detectable sources (3D), their corresponding frequencies (3D), their corresponding distance upper limit and their strain (if included). It also returns the number of the realization and a 1d array of a detection's frequency. '''
    FilteredLumdist = np.zeros(np.shape(d))
    FilteredFreqs = np.zeros(np.shape(d))
    Detectable_SMBHB_strain = np.zeros(np.shape(d))
    corresponding_upper_limit = np.zeros(np.shape(d))
    realization_number = []
    frequency_of_SMBHB = []
    for i in np.arange(np.shape(d)[0]): #Iterates over each frequency 
        for j in np.arange(np.shape(d)[1]): #Iterates over each realization 
            for k in np.arange(np.shape(d)[2]): #Iterates over each loudest SMBHB
                if d[i,j,k] < u[i,j,k]:
                    FilteredLumdist[i,j,k] = d[i,j,k]
                    FilteredFreqs[i,j,k] = f[i]
                    corresponding_upper_limit[i,j,k] = u[i,j,k]
                    realization_number.append(j)
                    frequency_of_SMBHB.append(f[i])
                    if h != None:
                        Detectable_SMBHB_strain[i,j,k] = h[i,j,k]
    if h != None:
        return FilteredLumdist, FilteredFreqs, corresponding_upper_limit, Detectable_SMBHB_strain, np.array(realization_number), np.array(frequency_of_SMBHB)
    else:
        return FilteredLumdist, FilteredFreqs, corresponding_upper_limit, np.array(realization_number), np.array(frequency_of_SMBHB)

def Detectable_Strain(f,h,u):
    Filteredstrain = np.zeros(np.shape(h))
    FilteredFreqs = np.zeros(np.shape(h))
    corresponding_upper_limit = np.zeros(np.shape(h))
    realization_number = []
    frequency_of_SMBHB = []
    for i in np.arange(np.shape(h)[0]): #Iterates over each frequency
        for j in np.arange(np.shape(h)[1]): #Iterates over each realization
            for k in np.arange(np.shape(h)[2]): #Iterates over each loudest binary
                if h[i,j,k] > u[i]:
                    Filteredstrain[i,j,k] = h[i,j,k]
                    FilteredFreqs[i][j][k] = f[i]
                    corresponding_upper_limit[i][j][k] = u[i]
                    realization_number.append(j)
                    frequency_of_SMBHB.append(f[i])
    return Filteredstrain, FilteredFreqs, corresponding_upper_limit, np.array(realization_number), np.array(frequency_of_SMBHB)





def Single_Par_Variation_GW(fobs_edges,ranges,func=None,par=None,shape=30,NUM_REALS=100,NUM_LOUDEST=3):
    '''Input: Frequency edges, 1D array with range of single parameter that should be varied. Output: GW information where param is varied.'''
    sam = []
    hc_ss, hc_bg, par_ss, par_bg = [], [], [], []    
    FidPars = [-2.5,11.5,8.3,0.45,5.55,-0.75] #2309.07227
    
    if func == 'gsmf':
        if par == 'phi0':
            for i,v in enumerate(ranges):
                sam.append(sams.Semi_Analytic_Model(shape=shape , gsmf=sams.components.GSMF_Schechter(phi0=v,mchar0_log10=FidPars[1]) , mmbulge=host_relations.MMBulge_Standard(mamp_log10=FidPars[2],scatter_dex=FidPars[3])))
        elif par == 'm0':
            for i,v in enumerate(ranges):
                sam.append(sams.Semi_Analytic_Model(shape=shape , gsmf=sams.components.GSMF_Schechter(phi0=FidPars[0],mchar0_log10=v) , mmbulge=host_relations.MMBulge_Standard(mamp_log10=FidPars[2],scatter_dex=FidPars[3])))
    
    
    elif func == 'mmbulge':
        if par == 'mass_norm':
            for i,v in enumerate(ranges):
                sam.append(sams.Semi_Analytic_Model(shape=shape , gsmf=sams.components.GSMF_Schechter(phi0=FidPars[0],mchar0_log10=FidPars[1]) ,  mmbulge=host_relations.MMBulge_Standard(mamp_log10=v,scatter_dex=FidPars[3]) ))
        elif par == 'scatter':
            for i,v in enumerate(ranges):
                sam.append(sams.Semi_Analytic_Model(shape=shape , gsmf=sams.components.GSMF_Schechter(phi0=FidPars[0],mchar0_log10=FidPars[1]), mmbulge=host_relations.MMBulge_Standard(mamp_log10=FidPars[2],scatter_dex=v)))
    
    
    elif func == 'hardening':
        for i,v in enumerate(ranges):
                sam.append(sams.Semi_Analytic_Model(shape=shape , gsmf=sams.components.GSMF_Schechter(phi0=FidPars[0],mchar0_log10=FidPars[1]), mmbulge=host_relations.MMBulge_Standard(mamp_log10=FidPars[2],scatter_dex=FidPars[3]))) #Effects of Hardening considered on GWB
   
    
    else:
        raise ValueError('Please input a valid function and parameter you want to vary.\n'
                         'Currently accepted functions and parameter pairs are (gsmf, (phi0 or m0)), (mmbulge, (mass_norm or scatter)) or (hardening, (tau, gamma_inner)).')
    
    if func == 'hardening':
        if par == 'gamma_inner':
            for i,v in enumerate(ranges):
                saves = sam[i].gwb(fobs_edges,hard=hardening.Fixed_Time_2PL_SAM(sam=sam[i],time=FidPars[4]*GYR,gamma_inner=v), realize=NUM_REALS, loudest=NUM_LOUDEST, params=True)
                hc_ss.append(saves[0])
                hc_bg.append(saves[1])
                par_ss.append(saves[2])
                par_bg.append(saves[3])
                
        if par == 'tau':
            for i,v in enumerate(ranges):
                saves = sam[i].gwb(fobs_edges,hard=hardening.Fixed_Time_2PL_SAM(sam=sam[i],time=v*GYR,gamma_inner=FidPars[5]), realize=NUM_REALS, loudest=NUM_LOUDEST, params=True)
                hc_ss.append(saves[0])
                hc_bg.append(saves[1])
                par_ss.append(saves[2])
                par_bg.append(saves[3])

                
    elif func != 'hardening':
        for i in np.arange(len(ranges)):
            saves = sam[i].gwb(fobs_edges,hard=hardening.Fixed_Time_2PL_SAM(sam=sam[i],time=FidPars[4]*GYR,gamma_inner=FidPars[5]), realize=NUM_REALS, loudest=NUM_LOUDEST, params=True)
            hc_ss.append(saves[0])
            hc_bg.append(saves[1])
            par_ss.append(saves[2])
            par_bg.append(saves[3])
            
    return hc_ss, hc_bg, par_ss, par_bg



def Double_Par_Variation_GW(fobs_edges,ranges,shape=30,NUM_REALS=100,NUM_LOUDEST=3):
    FidPars = dict([('psi0',-2.5),('m0',11.5),('mass_norm',8.3),('scatter',0.45),('tau',5.55),('gamma_inner',-0.75)])
    v0, v1 = list(ranges.values())
    k0, k1 = list(ranges.keys())
    l0, l1 = len(v0), len(v1)
    if k0 == k1:
      raise ValueError("Please vary two distinct parameters")  
    elif not isinstance(ranges,dict) or len(ranges) != 2:
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
        #Set each logging array to be 1 for the varied parameter
        log0[list(FidPars).index(k0)] = 1 
        log1[list(FidPars).index(k1)] = 1
                        
        for i,v in enumerate(v0): #Iterate over the range of the first variable
            print(f"GW single source strain calculation is {i/l0*100}% done.")
            for j,w in enumerate(v1): #Iterate over the range of the second variable
                pn = []
                for k,x in enumerate(FidParsl):
                    pn.append(x*(v/x)**(log0[k])*(w/x)**(log1[k])) #Gives a parameter that is not set to the fiducial value only if set in the ranges array
                sam[i,j] = sams.Semi_Analytic_Model(shape=shape , gsmf=sams.components.GSMF_Schechter(phi0=pn[0],mchar0_log10=pn[1]) , mmbulge=host_relations.MMBulge_Standard(mamp_log10=pn[2],scatter_dex=pn[3]))
                hc_ss[i,j], hc_bg[i,j], par_ss[i,j], par_bg[i,j] = sam[i,j].gwb(fobs_edges,hard=hardening.Fixed_Time_2PL_SAM(sam=sam[i,j],time=pn[4]*GYR,gamma_inner=pn[5]), realize=NUM_REALS, loudest=NUM_LOUDEST, params=True)
        return hc_ss, hc_bg, par_ss, par_bg



def Likelihood2Pars(Succ_Real_N,NUM_REALS):
    """Takes as input the object containing the realization numbers with at least one detection for each parameter pair (N,M). \n
    Returns a 2D array containing the Likelihood values """
    c1_var = np.zeros(np.shape(Succ_Real_N),dtype='object')
    for i,v in enumerate(Succ_Real_N):
        for j,w in enumerate(v):
            c1_var[i,j] = np.histogram(w,bins=NUM_REALS)[0]
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
    return 1 - success1_var/total1_var
        