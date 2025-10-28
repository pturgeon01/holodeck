"""Compare multiple MBH binary SAMS."""

import os
import sys
import h5py
import numpy as np
import pickle

import astropy as ap
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm

import kalepy as kale
import kalepy.utils
import kalepy.plot

import holodeck as holo
import holodeck.sams
import holodeck.gravwaves
from holodeck import cosmo, utils, plot, discrete, sams, host_relations, _PATH_DATA
from holodeck import utils, log
from holodeck.constants import PC, MSOL, YR, MPC, GYR, SPLC

from pathlib import Path


# ---- Define filepath containing simulation galaxy merger data files ----#
# ---- (if using files not in _PATH_DATA) ---- #
_HOME_PATH = Path('~/').expanduser()
p = os.path.join(_HOME_PATH, 'cosmo_sim_merger_data')
if os.path.exists(p):
    _SIM_MERGER_PATH = p
else:
    p = os.path.join(_HOME_PATH, 'nanograv/cosmo_sim_merger_data')
    if os.path.exists(p):
        _SIM_MERGER_PATH = p
    else:
        _SIM_MERGER_PATH = _PATH_DATA
#print(f"{_SIM_MERGER_PATH=}")
# ------------------------------------------------------------------------ #

class Test_SAM:
    
    def __init__(self, model_type='old', nreals=10, nloud=5, gpf_flag=0,
                 skip_evo=False, bfrac=None, 
                 hard_t=None, hard_ai=None, hard_rc=None, 
                 hard_nin=None, hard_nout=None, gsmf_flag = None):

        self.model_type = model_type
        self.nreals = nreals
        self.nloud = nloud
        self.gpf_flag = gpf_flag
        
        self.skip_evo = skip_evo
        self.bfrac = None
        print(f"\nCreating Test_SAM class instance with model_type={self.model_type}")

        if model_type == 'grid':
            if None in (hard_t, hard_ai, hard_rc, hard_nin, hard_nout, gsmf_flag):
                raise ValueError(f"cannot set grid elem for sam params if any keyword is None.")
                
            self.set_sam_params_grid(tau=hard_t, ai=hard_ai, rc=hard_rc, 
                                     nin=hard_nin, nout=hard_nout, mf = gsmf_flag)
        else:
            self.set_sam_params_manual(tau=hard_t)

        self.nfreqs = self.PARS['freqs'].size
        print(f"{self.nfreqs=}")
    
        if gpf_flag:
            print(f"creating SAM using GPF with {self.model_type=}...")
            self.lbl = model_type+'_gpf'
            self.sam = sams.Semi_Analytic_Model(gsmf = self.PARS['gsmf'], 
                                                gpf = sams.GPF_Power_Law())
        else:
            print(f"creating SAM using GMR with {self.model_type=}...")
            self.lbl = model_type+'_gmr'
            self.sam = sams.Semi_Analytic_Model(gsmf = self.PARS['gsmf'], gpf = None)

        
        print(f"    ...calculating hardening for GPF SAM with {self.model_type=}")
        self.hard = holo.hardening.Fixed_Time_2PL_SAM(self.sam, self.PARS['hard_time']*GYR, 
                                                      sepa_init = self.PARS['hard_sepa_init']*PC,
                                                      rchar = self.PARS['hard_rchar']*PC,
                                                      gamma_inner = self.PARS['hard_gamma_inner'],
                                                      gamma_outer = self.PARS['hard_gamma_outer'],
                                                     )

        ### ***NOTE*** gwb() allows for including pars of loud sources (unlike gwb_new()):
        print("    ...creating gwb for SAM")
        self.gwb_sam = self.sam.gwb(self.PARS['freqs_edges'], self.hard,
                                    realize=self.PARS['NREALS'], 
                                    loudest=self.PARS['NLOUD'], params=True)  


            
    def set_sam_params_manual(self, tau=None):

        if tau is None:
            raise ValueError("must choose a numerical value of keyword tau (in Gyr)!")
        
        # ---- Define the GWB frequencies and other key model params
        if self.model_type == 'old':
            self.PARS = dict(
                desc='LB old',
                hard_sepa_init=1e4,     # [pc]
                hard_rchar=10.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+1.5,
                gsmf = holo.sams.GSMF_Schechter()
            )
        elif self.model_type == 'old_2s':
            self.PARS = dict(
                desc='LB old + 2-Schechter',
                hard_sepa_init=1e4,     # [pc]
                hard_rchar=10.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+1.5,
                gsmf = holo.sams.GSMF_Double_Schechter()
            )
        elif self.model_type == 'old_rc100':
            self.PARS = dict(
                desc='LB old + rchar=100',
                hard_sepa_init=1e4,     # [pc]
                hard_rchar=100.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+1.5,
                gsmf = holo.sams.GSMF_Schechter()
            )
        elif self.model_type == 'ph15':
            self.PARS = dict(
                desc='15yr Phenom',
                hard_sepa_init=1e4,     # [pc]
                hard_rchar=100.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+2.5,
                gsmf = holo.sams.GSMF_Schechter()
            )
        elif self.model_type == 'astr':
            self.PARS = dict(
                desc='Astro Strong',
                hard_sepa_init=1e4,     # [pc]
                hard_rchar=10.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+2.5,
                gsmf = holo.sams.GSMF_Double_Schechter()
            )
        elif self.model_type == 'astr_nuo0':
            self.PARS = dict(
                desc='Astro Strong w/ nu_outer=0',
                hard_sepa_init=1e4,     # [pc]
                hard_rchar=10.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=0.0,
                gsmf = holo.sams.GSMF_Double_Schechter()
            )
        elif self.model_type == 'astr_rc100':
            self.PARS = dict(
                desc='Astro Strong w/ rchar=100',
                hard_sepa_init=1e4,     # [pc]
                hard_rchar=100.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+2.5,
                gsmf = holo.sams.GSMF_Double_Schechter()
            )
        else:
            modlist = ['old', 'old_2s', 'old_rc100', 'ph15', 
                       'astr', 'astr_nuo0', 'astr_rc100']

            raise ValueError(f"{self.model_type=} is not defined. Options are {[m for m in modlist]}.")

        # define the fixed inspiral timescale from input keyword
        self.PARS["hard_time"] = tau    # [Gyr]

        # add other params that will stay the same between model types
        freqs, freqs_edges = utils.pta_freqs()
        self.PARS["freqs"] = freqs
        self.PARS["freqs_edges"] = freqs_edges
        self.PARS["NREALS"] = self.nreals
        self.PARS["NLOUD"] = self.nloud

        print(f"Set SAM params manually for {self.model_type=}.")
    

    def set_sam_params_grid(self, tau=None, ai=None, rc=None, 
                            nin=None, nout=None, mf = None):
        
        if mf == 1:
            _gsmf = holo.sams.GSMF_Schechter()
        elif mf == 2:
            _gsmf = holo.sams.GSMF_Double_Schechter()
        else:
            raise ValueError(f"invalid gsmf_flag: {mf}. must be 1 or 2.")

        self.PARS = dict(
            desc=(f't{tau}ai{ai}rc{rc}nin{nin}nout{nout}gsmf{mf}'),
            hard_time=tau,          # [Gyr]
            hard_sepa_init=ai,     # [pc]
            hard_rchar=rc,        # [pc]
            hard_gamma_inner=nin,
            hard_gamma_outer=nout,
            gsmf = _gsmf
        )
        
        # add other params that will stay the same between model types
        freqs, freqs_edges = utils.pta_freqs()
        self.PARS["freqs"] = freqs
        self.PARS["freqs_edges"] = freqs_edges
        self.PARS["NREALS"] = self.nreals
        self.PARS["NLOUD"] = self.nloud

        print(f"Set SAM param grid elem for {self.PARS['desc']}.")
        

        
def create_sams(nreals=5, nloud=5, fpath=_PATH_DATA, suite_type='grid',
                ai=None, tau=None, gsmf=None, gpfflag=None, 
                pickle_sams=True, pickle_name=None):

    all_sams = []
    
    if suite_type == 'grid':
        for t in [1.0,3.0]:
            for rc in [10.0, 100.0]:
                for nin in [-2.0, -0.5]:
                    for nout in [0.0, +2.5]:
                        #for mf in [1,2]:

                        s = Test_SAM(model_type='grid', nreals=nreals, nloud=nloud, 
                                     hard_t=t, hard_ai=ai, hard_rc=rc, 
                                     hard_nin=nin, hard_nout=nout,
                                     gsmf_flag = gsmf, gpf_flag=gpfflag)

                        all_sams = all_sams + [s]
                            
        if pickle_sams and pickle_name is None:
            gpf_str = ['gmr', 'gpf']
            pickle_name = f'wide_grid_{gpf_str[gpfflag]}_ai{ai}_{gsmf}schech'

    elif suite_type == 'manual':
        test_model_list = [
                           'old',
                           'old_2s', 
                           'old_rc100', 
                           'ph15', 
                           'astr', 
                           'astr_nuo0', 
                           'astr_rc100',
                          ]
    
        for mod in test_model_list:
            if tau is None:
                print(f'WARNING: keyword `tau` set to None. Assigning hard tscale tau=1.0 Gyr.')
                tau = 1.0
                
            print(f'Creating test SAM for model_type {mod} with gpf_flag={gpfflag} & tau={tau}.')
            s = Test_SAM(model_type=mod, nreals=nreals, nloud=nloud, gpf_flag=gpfflag, hard_t=tau)

            all_sams = all_sams + [s]
                
        if pickle_sams and pickle_name is None:
            gpf_str = ['gmr', 'gpf']
            pickle_name = f'manual_moddefs_{gpf_str[gpfflag]}_tau{tau}'

    if pickle_sams:
        nfreqs = all_sams[0].PARS['freqs'].size
        pkl_fname = f"test_sam_nfreqs{nfreqs}_nreals{nreals}_nloud{nloud}_{pickle_name}.pkl"

        print(f"creating pkl file: {pkl_fname}")
        with open(f"{fpath}/{pkl_fname}", "wb") as f:
            pickle.dump(all_sams, f)

    return all_sams 


if __name__ == '__main__':

    suite_type = 'manual'
    #suite_type = 'grid'
    
    if suite_type == 'grid':
    
        # ---- 'GRID' SETUP ----# 
        if len(sys.argv)>3:
            NREALS = int(sys.argv[1])
            NLOUD = int(sys.argv[2])
            GPFFLAG = int(sys.argv[3])
            GSMF = int(sys.argv[4])
            if len(sys.argv)>5:
                print("Too many command line args ({sys.argv}).")
                sys.exit()

        else:
            print("expecting 4 command line args: NREALS, NLOUD, GPFFLAG, GSMF.")
            sys.exit()

        tmp = create_sams(nreals=NREALS, nloud=NLOUD, fpath=_PATH_DATA, 
                          suite_type='grid', ai=1e4, gsmf=GSMF, gpfflag=GPFFLAG,
                          pickle_sams=True, pickle_name=None)

    elif suite_type == 'manual':

        # ---- 'MANUAL' SETUP ----#         
        if len(sys.argv)>3:
            NREALS = int(sys.argv[1])
            NLOUD = int(sys.argv[2])
            GPFFLAG = int(sys.argv[3])
            TAU = float(sys.argv[4])
            if len(sys.argv)>5:
                print("Too many command line args ({sys.argv}).")
                sys.exit()

        else:
            print("expecting 4 command line args: NREALS, NLOUD, GPFFLAG, TAU.")
            sys.exit()

        tmp = create_sams(nreals=NREALS, nloud=NLOUD, fpath=_PATH_DATA, 
                          suite_type='manual', gpfflag=GPFFLAG, tau=TAU,
                          pickle_sams=True, pickle_name=None)

    else:
        raise ValueError("`suite_type` must be 'grid' or 'manual'.")