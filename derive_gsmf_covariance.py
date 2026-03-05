import numpy as np

# Redshift values for anchor points
ZVALS = [0.2, 1.6, 3.0]

def get_cs(yy):
    z1, z2, z3 = ZVALS
    
    # Input yy can be (3, N) or (3,)
    if yy.ndim == 1:
        y1, y2, y3 = yy
    else:
        y1, y2, y3 = yy

    # calculate 'c2' - z^2 prefactor
    tt = (z1 - z3) / (z2 - z1)
    denom = (z2**2 - z1**2) * tt
    denom = z3**2 - z1**2 + denom
    numer = (y3 - y1) + (y2 - y1) * tt
    c2 = numer / denom

    # calculate 'c1' - z prefactor
    c1 = y2 - y1 - c2*(z2**2 - z1**2)
    c1 /= (z2 - z1)

    # calculate 'c0' - constant term
    c0 = y1 - c2*z1**2 - c1*z1
    
    return np.array([c0, c1, c2])

# Anchor point data from Leja+2020 Fig 3 (via notebook)
pars = {
    'logphi1': [-2.44, -3.08, -4.14],
    'logphi1_err': [0.02, 0.03, 0.1],
    'logphi2': [-2.89, -3.29, -3.51],
    'logphi2_err': [0.04, 0.03, 0.03],
    'logmstar': [10.79, 10.88, 10.84],
    'logmstar_err': [0.02, 0.02, 0.04],
}

# Number of samples for propagation
NUM = 100000

def propagate_uncertainties(par_name):
    yave = np.array(pars[par_name])
    yerr = np.array(pars[par_name + "_err"])
    
    # Sample anchor points (assuming independence for now, as in notebook)
    yy_samples = np.random.normal(yave[:, np.newaxis], yerr[:, np.newaxis], size=(3, NUM))
    
    # Transform to coefficients
    cs_samples = get_cs(yy_samples)
    
    means = np.mean(cs_samples, axis=1)
    sigmas = np.std(cs_samples, axis=1)
    return means, sigmas, cs_samples

all_means = []
all_sigmas = []
all_samples = []

for name in ['logphi1', 'logphi2', 'logmstar']:
    m, s, samp = propagate_uncertainties(name)
    all_means.extend(m)
    all_sigmas.extend(s)
    all_samples.append(samp)

# Add alpha1, alpha2
# From notebook: means = [... , -0.40, -1.53]
# sigmas = [... , 0.07, 0.015]
all_means.extend([-0.40, -1.53])
all_sigmas.extend([0.07, 0.015])

# Final means and sigmas
final_means = np.array(all_means)
final_sigmas = np.array(all_sigmas)

print("Derived Means:")
print(repr(final_means))
print("\nDerived Sigmas:")
print(repr(final_sigmas))

# Now construct the correlation matrix rho from notebook
n = 11
rho = np.eye(n)
rho[3, 9]  =  0.85  # log(phi2)c0 vs alpha1
rho[4, 5]  =  0.90  # log(phi2)c1 vs log(phi2)c2
rho[9, 10] =  0.60  # alpha1 vs alpha2
rho[6, 9]  = -0.70  # log(M*)c0 vs alpha1
rho[0, 3]  = -0.40  # log(phi1)c0 vs log(phi2)c0
rho[2, 4]  = -0.50  # log(phi1)c2 vs log(phi2)c1

rho = (rho + rho.T) - np.eye(n)

# Construct Covariance
cov = np.outer(final_sigmas, final_sigmas) * rho

def repair_covariance(m):
    m = (m + m.T) / 2
    eigval, eigvec = np.linalg.eigh(m)
    eigval = np.maximum(eigval, 1e-10)
    return eigvec @ np.diag(eigval) @ eigvec.T

cov_fixed = repair_covariance(cov)

print("\nFinal Fixed Covariance Matrix:")
print(repr(cov_fixed))
