from sympy.combinatorics.permutations import Permutation
from numpy.random import uniform, normal

import pickle
import sys
import os
import time
from tqdm import tqdm
from tn import *

######################################
################ TI ##################
######################################

def one_copy_projector_ti_subspace(local_dimension, number_sites):
    """ returns r**number_sites where r is the effective local dimension """ 
    symmetry = np.concatenate([np.arange(number_sites, dtype=int), np.arange(number_sites, dtype=int)])
    permutation_list = [symmetry[i : i + number_sites] for i in range(number_sites)]
    length_permutation_list = len(permutation_list)
    cycle_counts = np.array([Permutation(permutation_list[i]).cycles for i in range(length_permutation_list)], dtype=int)
    return_value = np.sum(float(local_dimension) ** cycle_counts) / len(permutation_list)
    return return_value

class TI_MPS(MPS):
    def __init__(self, site, num_sites):
        super().__init__([site] * num_sites)
    
    def overlap(self, other, scaled=False): 
        """ if scaled, multiplies every transfer matrix by sqrt(r) where d is the effective local dimension """
        if scaled:
            scale = np.sqrt(one_copy_projector_ti_subspace(self.local_dim, self.num_sites)**(1/self.num_sites)) 
        else: 
            scale = 1.0
            
        tensor = ncon((self.sites[0], other.sites[0].conj()), ([-1,-3,1],[-2,-4,1])) * scale
        for i in range(1,self.num_sites):
            tensor = ncon((tensor, self.sites[i]), ([-1,-2,1,-3],[1,-4,-5]))
            tensor = ncon((tensor, other.sites[i].conj()), ([-1,-2,1,-3,2],[1,-4,2])) * scale
        return np.trace(tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2] * tensor.shape[3]))

######################################
############### HAAR #################
######################################

def haar_purity(num_sites, k_copies, local_dim=2, ti=False, scaled=False): 
    """ if scaled, returns purity * r**(num_sites*k_copies) where r is the effective local dimension """
    q = one_copy_projector_ti_subspace(local_dim, num_sites) if ti else local_dim**num_sites
    q_norm = q if scaled else 1
    factors = [(q+i)/q_norm for i in range(k_copies)]
    return np.math.factorial(k_copies)/np.prod(factors)

def haar_random_isometry(l_chi, r_chi=None, local_dim=2):
    r_chi = l_chi if r_chi is None else r_chi
    size = local_dim * l_chi
    temp = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    temp = temp + temp.conj().T 
    return np.linalg.eigh(temp)[1][:, :r_chi]

#######################################
############## RMPS ###################
#######################################

def open_rmps_staggered(num_sites, chi, local_dim=2, distrib='isometric', param=1.0):
    sites = []
    for i in range(num_sites): 
        l_chi = min(local_dim**i, chi, local_dim**(num_sites-i))
        r_chi = min(local_dim**(i+1), chi, local_dim**(num_sites-1-i))
        shape = (l_chi, r_chi, local_dim)
        
        if distrib == 'isometric':
            site = haar_random_isometry(*shape).reshape(l_chi, local_dim, r_chi).transpose(0,2,1)
        elif distrib == 'uniform': 
            site = uniform(-1, 1, shape) + 1.j*uniform(-1, 1, shape)
        elif distrib == 'gaussian':
            site = normal(0, param, shape) + 1.j*normal(0, param, shape)
        else: 
            raise ValueError(f"{distrib} is not a valid distribution for site tensors")
        sites.append(site)
        
    mps = MPS(sites)
    if distrib != 'isometric': 
        mps.normalize()
    return mps 

def open_rmps_even(num_sites, chi, local_dim=2, distrib='isometric', param=1.0):
    sites = []
    for _ in range(num_sites):
        if distrib == 'isometric':
            site = haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1) 
        elif distrib == 'uniform':
            site = uniform(-1, 1, (chi, chi, local_dim)) + 1.j * uniform(-1, 1, (chi, chi, local_dim))
        elif distrib == 'gaussian':
            site = normal(0, param, (chi, chi, local_dim)) + 1.j * normal(0, param, (chi, chi, local_dim))
        else: 
            raise ValueError(f"{distrib} is not a valid distribution for site tensors")
        sites.append(site)

    edge = np.zeros(chi)
    edge[0] = 1.0
    sites[0] = ncon((edge, sites[0]), ([1], [1,-1,-2]))[np.newaxis, :, :]
    sites[-1] = ncon((sites[-1], edge), ([-1,1,-2], [1]))[:, np.newaxis, :]
    mps = MPS(sites)
    mps.normalize()
    return mps
    
def periodic_rmps(num_sites, chi, local_dim=2, distrib="isometric", param=1.0): 
    sites = [haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1) 
             for _ in range(num_sites)]
    mps = MPS(sites)
    mps.normalize()
    return mps

def ti_rmps(num_sites, chi, local_dim=2, distrib="isometric", param=1.0):
    site = haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1)
    mps = TI_MPS(site, num_sites)
    mps.normalize()
    return mps

#######################################
############# PURITY ##################
#######################################

def rmps_purity(num_sites, k_copies, chi, samples=5000, local_dim=2, func=periodic_rmps, 
                distrib='isometric', param=1.0, quiet=True):
    """ generates samples and then computes purity; meant for interactive use """
    avg_purity = 0.0
    r = tqdm(range(samples)) if not quiet else range(samples)
    for _ in r:  
        rmps_a = func(num_sites, chi, local_dim, distrib=distrib, param=param)
        rmps_b = func(num_sites, chi, local_dim, distrib=distrib, param=param)           
        avg_purity += np.abs(rmps_a.overlap(rmps_b))**(2*k_copies)
    
    avg_purity /= samples
    return avg_purity

def trace_samples(num_sites, chi, num_samples=5000, local_dim=2, func=periodic_rmps, distrib='isometric', 
                                                      param=1.0, quiet=True, scaled=False):
    """ helper function for generate_samples """
    r = tqdm(range(num_samples)) if not quiet else range(num_samples)
    samples = []
    for _ in r: 
        rmps_a = func(num_sites, chi, local_dim, distrib=distrib)
        rmps_b = func(num_sites, chi, local_dim)
        trace = np.abs(rmps_a.overlap(rmps_b, scaled=scaled))
        samples.append(trace)
    return samples

def generate_samples(n_list, chi_list, filename, num_samples=5000, func=periodic_rmps, distrib='isometric', param=1.0):
    """ generates local_dim=2 scaled samples; meant for cluster """
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {}
    
    for i,(n,chi) in tqdm(enumerate(zip(n_list,chi_list))): 
        if (n,chi) not in data_dict.keys():
            data_dict[(n,chi)] = trace_samples(n,chi,num_samples, scaled=True, func=func, distrib=distrib, param=param)
            with open(filename, "wb") as f: 
                pickle.dump(data_dict, f)

    return data

def samples_to_purity(samples, k_copies): 
    """ turns samples to purity """
    return np.sum(np.array(samples)**(2*k_copies)) / len(samples)

def main():
    min_n, max_n, n_incr, num_samples, mps_type, chi = sys.argv[1:7]
    filename = [mps_type[:1], chi]
    
    if len(sys.argv) > 7: 
        distrib = sys.argv[7]
        filename.insert(1, distrib)
    else: 
        distrib = 'isometric'
        
    if len(sys.argv) > 8: 
        param = sys.argv[8]
        filename.insert(2, param)
    else: 
        param = 1.0
                       
    filename = "_".join(filename) + ".pickle"
    n_list = np.arange(int(min_n), int(max_n)+1, int(n_incr))
    
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
    elif mps_type == "open staggered":
        func = open_rmps_staggered
    elif mps_type == "open even":
        func = open_rmps_even
    elif mps_type == "ti": 
        func = ti_rmps
    else:
        raise ValueError(f"{mps_type} is not a valid option for mps_type")

    
    t1 = time.time()
    generate_samples(n_list, chi_list, filename, int(num_samples), func=func, distrib=distrib, param=float(param))
    t2 = time.time()
    print(f"generated {filename} in {(t2-t1)/60} minutes")
    
if __name__ == "__main__":
    main()