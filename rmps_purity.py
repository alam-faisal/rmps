from scipy.special import comb
import pickle
import sys
import time
from tqdm import tqdm
from tn import *

"""
if scaled = False, purity corresponds to two-norm error 
if scaled = True, every transfer matrix is multiplied by sqrt(p) and **purity** corresponds to one-norm error
"""

######################################
############### HAAR #################
######################################

def haar_purity(num_sites, k_copies, local_dim=2, scaled=False): 
    q = local_dim**num_sites if scaled else 1
    factors = [(local_dim**num_sites + k_copies - 1 - i)/q for i in range(k_copies)]
    return np.math.factorial(k_copies)/np.prod(factors)

def ti_purity(num_sites, k_copies, local_dim=2, scaled=False): 
    raise NotImplementedError("call Shivan")

def haar_random_isometry(l_chi, r_chi=None, local_dim=2):
    r_chi = l_chi if r_chi is None else r_chi
    size = local_dim * l_chi
    temp = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    temp = temp + temp.conj().T 
    return np.linalg.eigh(temp)[1][:, :r_chi]

#######################################
############## RMPS ###################
#######################################

def open_rmps(num_sites, chi, local_dim=2):
    sites = []
    for i in range(num_sites): 
        l_chi = min(local_dim**i, chi, local_dim**(num_sites-i))
        r_chi = min(local_dim**(i+1), chi, local_dim**(num_sites-1-i))
        site = haar_random_isometry(l_chi, r_chi, local_dim).reshape(l_chi, local_dim, r_chi).transpose(0,2,1)
        sites.append(site)
    return MPS(sites)

def periodic_rmps(num_sites, chi, local_dim=2): 
    sites = [haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1) 
             for _ in range(num_sites)]
    mps = MPS(sites)
    mps.normalize()
    return mps

def ti_rmps(num_sites, chi, local_dim=2):
    sites = [haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1)] * num_sites
    mps = MPS(sites)
    mps.normalize()
    return mps

#######################################
############# PURITY ##################
#######################################

def rmps_purity(num_sites, k_copies, chi, samples=5000, local_dim=2, func=periodic_rmps, quiet=True):
    """ generates samples and then computes purity; meant for interactive use """
    avg_purity = 0.0
    r = tqdm(range(samples)) if not quiet else range(samples)
    for _ in r:  
        rmps_a = func(num_sites, chi, local_dim)
        rmps_b = func(num_sites, chi, local_dim)           
        avg_purity += np.abs(rmps_a.overlap(rmps_b))**(2*k_copies)
    
    avg_purity /= samples
    return avg_purity

def trace_samples(num_sites, chi, num_samples=5000, local_dim=2, func=periodic_rmps, quiet=True, scaled=False):
    """ helper function for generate_samples """
    r = tqdm(range(num_samples)) if not quiet else range(num_samples)
    samples = []
    for _ in r: 
        rmps_a = func(num_sites, chi, local_dim)
        rmps_b = func(num_sites, chi, local_dim)
        trace = np.abs(rmps_a.overlap(rmps_b, scaled=scaled))
        samples.append(trace)
    return samples

def generate_samples(n_list, chi_list, filename, num_samples=5000, func=periodic_rmps):
    """ generates local_dim=2 scaled samples; meant for cluster """
    data = np.zeros((len(n_list), num_samples))
    for i,(n,chi) in tqdm(enumerate(zip(n_list,chi_list))): 
        data[i,:] = np.array(trace_samples(n,chi,num_samples, scaled=True, func=func))
        with open(filename, "wb") as f: 
            pickle.dump(data, f)

    return data

def samples_to_purity(samples, k_copies): 
    """ turns samples to purity """
    return np.sum(np.array(samples)**(2*k_copies)) / len(samples)

def main():
    max_n, n_incr, num_samples, mps_type, chi = sys.argv[1:]
    n_list = np.arange(4, int(max_n)+1, int(n_incr))
    
    if chi == "linear":
        chi_list = [n for n in n_list]
    elif chi == "superlinear":
        chi_list = [int(n**1.5) for n in n_list]
    elif chi == "quadratic":
        chi_list = [int(n**2 / 2) for n in n_list]
    elif chi == "cubic":
        chi_list = [int(n**3 / 3) for n in n_list]
    else:
        chi_list = [int(chi) for n in n_list]
        
    if mps_type == "periodic": 
        func = periodic_rmps
    elif mps_type == "open":
        func = open_rmps
    elif mps_type == "ti": 
        func = ti_rmps
    else:
        raise ValueError(f"{mps_type} is not a valid option for mps_type")

    filename = "_".join([mps_type, chi]) + ".pickle"
    t1 = time.time()
    generate_samples(n_list, chi_list, filename, int(num_samples), func=func)
    t2 = time.time()
    print(f"generated {filename} in {(t2-t1)/60} minutes")
    
if __name__ == "__main__":
    main()