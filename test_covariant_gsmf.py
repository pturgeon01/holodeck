import numpy as np
import holodeck as holo
from holodeck.librarian import param_spaces

def test_covariant_gsmf():
    print("Initializing PS_Astro_Strong_Covariant_GSMF...")
    ps = param_spaces.PS_Astro_Strong_Covariant_GSMF(nsamples=10)
    
    print("Drawing samples...")
    # params should be (nsamples, nparams)
    params = ps.extrema
    print(f"Extrema shape: {params.shape}")
    
    # Draw from the distribution
    samples = ps.param_samples
    print(f"Samples shape: {samples.shape}")
    print(f"First sample: {samples[0]}")
    
    expected_means = np.array([-2.38282317, -0.26438520, -0.10710090,
                              -2.81825115, -0.36823094,  0.04588269,
                              10.76657073,  0.12372112, -0.03305944, 
                              -0.40, -1.53])
    
    print(f"Expected means first 3: {expected_means[:3]}")
    print(f"Sample 0 first 3: {samples[0, :3]}")

    print("\nInitializing SAM with first sample...")
    # model_for_params expects a dictionary
    param_dict = dict(zip(ps.param_names, samples[0]))
    sam, hard = ps.model_for_params(param_dict)
    print("SAM and Hardening initialized successfully.")
    
    # Calculate GWB to ensure everything is connected
    fobs = np.logspace(-9, -7, 10)
    gwb = sam.gwb(fobs)
    print(f"GWB calculated: {gwb}")

if __name__ == "__main__":
    test_covariant_gsmf()
