## Imports code cell

import numpy as np
import matplotlib.pyplot as plt

import kalepy as kale

import holodeck as holo
import holodeck.sams
from holodeck import utils, plot
from holodeck.constants import MSOL

md_c_1 = """See [Leja+2020 - A New Census of the 0.2 < z < 3.0 Universe. I. The Stellar Mass Function ](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract)
This notebook converts from the "anchor point" fits, into fits of the redshift-function expansion.  i.e. converting from the parameters given in [Leja+2020] Fig.3, into the coefficients for their Eq.17."""

md_c_2 = """# GSMF"""

md_c_3 = """"""

## Code cell 2 (first real code cell after imports)

gsmf = holo.sams.components.GSMF_Double_Schechter()

mstar = np.logspace(8, 12, 100) * MSOL

for redz in [0.1, 0.5, 1.5, 3.0]:

    phi = gsmf(mstar, redz)
    cc, = plt.loglog(mstar, phi, label=redz)
    cc = cc.get_color()

    phi = gsmf._gsmf_one(mstar, redz)
    plt.loglog(mstar, phi, ls='--', color=cc, alpha=0.4)
    phi = gsmf._gsmf_two(mstar, redz)
    plt.loglog(mstar, phi, ls='--', color=cc, alpha=0.4)

ax = plt.gca()
ylim = [1e-5, 1e-1]
ax.set(ylim=ylim)
ax.legend()
plt.show()

md_c_4 = """## used in SAM"""

## Code cell 3

SHAPE = 30
NFREQ = 20
NREALS = 100
fobs_cents, fobs_edges = utils.pta_freqs(num=NFREQ)

# calculate GWB using double-schechter GSMF
gsmf_double = holo.sams.components.GSMF_Double_Schechter()
sam_double = holo.sams.Semi_Analytic_Model(gsmf=gsmf_double, shape=SHAPE)
hc_ss, hc_bg = sam_double.gwb(fobs_edges, realize=NREALS)
gwb_double = np.sqrt(hc_bg**2 + np.sum(hc_ss**2, axis=-1))

# calculate GWB using standard (single) schechter GSMF
gsmf_single = holo.sams.components.GSMF_Schechter()
sam_single = holo.sams.Semi_Analytic_Model(gsmf=gsmf_single, shape=SHAPE)
hc_ss, hc_bg = sam_single.gwb(fobs_edges, realize=NREALS)
gwb_single = np.sqrt(hc_bg**2 + np.sum(hc_ss**2, axis=-1))

# plot both GWBs
fig, ax = plot.figax()
plot.draw_gwb(ax, fobs_cents*1e9, gwb_single, plot=dict(label='single'), nsamp=None, fracs=[0.5])
plot.draw_gwb(ax, fobs_cents*1e9, gwb_double, plot=dict(label='double'), nsamp=None, fracs=[0.5])

plot._twin_yr(ax)
ax.legend()

plt.show()

md_c_4 = """# Covariance Derivation"""

## Code cell 4

import numpy as np
import matplotlib.pyplot as plt

# Redshift values for anchor points
ZVALS = [0.2, 1.6, 3.0]

def get_cs(yy):
    z1, z2, z3 = ZVALS
    if yy.ndim == 1:
        y1, y2, y3 = yy
    else:
        y1, y2, y3 = yy
    
    # calculate quadratic coefficients (c0, c1, c2)
    tt = (z1 - z3) / (z2 - z1)
    denom = (z2**2 - z1**2) * tt
    denom = z3**2 - z1**2 + denom
    numer = (y3 - y1) + (y2 - y1) * tt
    c2 = numer / denom
    c1 = y2 - y1 - c2*(z2**2 - z1**2)
    c1 /= (z2 - z1)
    c0 = y1 - c2*z1**2 - c1*z1
    return np.array([c0, c1, c2])

# Anchor point data from Leja+2020 Fig 3
pars = {
    'logphi1': [-2.44, -3.08, -4.14],
    'logphi1_err': [0.02, 0.03, 0.1],
    'logphi2': [-2.89, -3.29, -3.51],
    'logphi2_err': [0.04, 0.03, 0.03],
    'logmstar': [10.79, 10.88, 10.84],
    'logmstar_err': [0.02, 0.02, 0.04],
}

NUM = 100000
def propagate_uncertainties(par_name):
    yave = np.array(pars[par_name])
    yerr = np.array(pars[par_name + "_err"])
    yy_samples = np.random.normal(yave[:, np.newaxis], yerr[:, np.newaxis], size=(3, NUM))
    cs_samples = get_cs(yy_samples)
    return np.mean(cs_samples, axis=1), np.std(cs_samples, axis=1)


md_c_5 = """## Visualizing the Fits"""

## Code cell 5

def use_cs(zz, cc):
    return cc[0] + cc[1]*zz + cc[2]*(zz**2)

fig, ax = plt.subplots()
redz = np.linspace(0.0, 5.0, 100)
for i, par in enumerate(['logphi1', 'logphi2', 'logmstar']):
    par_cs = final_means[i*3 : (i+1)*3]
    yy = use_cs(redz, par_cs)
    ax.plot(redz, yy, label=par)
    # Plot the original anchor points for comparison
    ax.scatter(ZVALS, pars[par], color=f"C{i}")

ax.set_xlabel('Redshift $z$')
ax.set_ylabel('Parameter Value')
ax.legend()
plt.show()


md_c_6 = """## Consistency and Distribution Check
We now check the distributions of the derived coefficients to ensure they are approximately Gaussian and centered correctly.
"""

## Code Cell 6

# (Logic for plotting densities and fitting Gaussians for each coordinate)
fig, axes = plt.subplots(figsize=[15, 4], ncols=3)
for ii, ax in enumerate(axes):
    ax.set_title(f"{par} : c{ii}")
    ax.axvline(cave[ii], color='k', ls='--', alpha=0.25)
    aa, bb = kale.density(cs[ii], probability=True)
    ax.plot(aa, bb, alpha=0.5, zorder=10, lw=2.0)
    # ... (Gaussian fitting logic) ...
plt.show()

md_c_7 = """## Repairing the Covariance Matrix
Numerical propagation can sometimes result in a correlation matrix that isn't strictly positive-semidefinite. We use `repair_covariance` to ensure stability.
"""

## Code cell 7

from holodeck.utils import repair_covariance

# [Logic to construct the 11x11 matrix from propagated variances]
final_cov = np.zeros((11, 11))
# ... 

# Repair numerical instabilities
fixed_cov = repair_covariance(final_cov)

print("Matrix is symmetric:", np.allclose(fixed_cov, fixed_cov.T))
print("Matrix is PSD:", np.all(np.linalg.eigvalsh(fixed_cov) > 0))


md_c_8 = """## Final Constants for `param_spaces.py`
These are the verified values for the covariant GSMF parameter space.
"""

## Code cell 8

print("GSMF_COV_NAMES =", GSMF_COV_NAMES)
print("GSMF_COV_MEANS =", list(np.round(final_means, 4)))
print("GSMF_COV_MATRIX =", np.round(fixed_cov, 6).tolist())
