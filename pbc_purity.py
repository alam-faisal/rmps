import numpy as np
from ncon import *
from tqdm import tqdm
from scipy.special import comb
import pickle
import sys
import time

def haar_random_isometry(chi, p=2):
    """ p is the local dimension """
    temp = np.random.rand(p * chi, p * chi) + 1j * np.random.rand(p * chi, p * chi)
    temp = temp + np.conjugate(np.transpose(temp))
    return np.linalg.eigh(temp)[1][:, :chi]

def periodic_rmps(n, chi, p=2, ti=False):
    """ 
    random MPS with period boundary conditions
    ti = True leads to translation invariant MPSs 
    """
    if ti: 
        tensor_list = [np.reshape(haar_random_isometry(chi, p), [chi, p, chi])]*n
    else: 
        tensor_list = [np.reshape(haar_random_isometry(chi, p), [chi, p, chi]) for _ in range(n)]
    
    prmps = np.array(tensor_list)
    prmps = prmps / np.sqrt(np.abs(overlap(prmps, prmps, chi, ti=ti)))**(1/n)
    return prmps  

"""
if scaled = False, purity corresponds to two-norm error 
if scaled = True, every transfer matrix is multiplied by sqrt(p) and **purity** corresponds to one-norm error
"""

def haar_purity(n, k_copies, p=2, scaled=False): 
    q = p**n if scaled else 1
    factors = [(p**n + k_copies - 1 - i)/q for i in range(k_copies)]
    return np.math.factorial(k_copies)/np.prod(factors)

def transfer_matrix(prmps1, prmps2, chi, i, scaled=False): 
    p = prmps1[0].shape[1]
    mat = np.reshape(ncon([prmps1[i], prmps2[i].conj()], [[-1, 1, -3], [-2, 1, -4]]), [chi**2, chi**2])
    return np.sqrt(p) * mat if scaled else mat

def transfer_matrices(prmps1, prmps2, chi, ti=False, scaled=False):
    n = len(prmps1)
    if ti: 
        return [transfer_matrix(prmps1, prmps2, chi, 0, scaled=scaled)]*n
    else: 
        return [transfer_matrix(prmps1, prmps2, chi, i, scaled=scaled) for i in range(n)]

def overlap(prmps1, prmps2, chi, ti=False, scaled=False):
    if ti: 
        mat = transfer_matrix(prmps1, prmps2, chi, 0, scaled=scaled)
        evals = np.linalg.eigvals(mat)
        return sum(evals**len(prmps1))
    else: 
        mat_list = transfer_matrices(prmps1, prmps2, chi, ti=ti, scaled=scaled)
        return np.trace(np.linalg.multi_dot(mat_list))
    
def trace_samples(n, chi, num_samples=5000, p=2, ti=False, quiet=True, check=False, scaled=False):
    r = tqdm(range(num_samples)) if not quiet else range(num_samples)
    samples = []
    for _ in r: 
        prmps_a = periodic_rmps(n, chi, p=p, ti=ti)
        prmps_b = periodic_rmps(n, chi, p=p, ti=ti)
        
        if check: 
            norm_a = overlap(prmps_a, prmps_a, chi)
            norm_b = overlap(prmps_b, prmps_b, chi)
            if np.abs(1 - norm_a) > 1.e-12:
                print(np.abs(1 - norm_a))
                raise ValueError("State not normalized!")

            if np.abs(1 - norm_b) > 1.e-12:
                print(np.abs(1 - norm_b))
                raise ValueError("State not normalized!")
                
        trace = np.abs(overlap(prmps_a, prmps_b, chi, scaled=scaled))
        samples.append(trace)
    return samples

def samples_to_purity(samples, k_copies): 
    return np.sum(np.array(samples)**(2*k_copies)) / len(samples)

def rmps_purity(n, k_copies, chi, samples=5000, p=2, ti=False, quiet=True, check=False):
    avg_purity = 0.0
    r = tqdm(range(samples)) if not quiet else range(samples)
    for _ in r:  
        prmps_a = periodic_rmps(n, chi, p=p, ti=ti)
        prmps_b = periodic_rmps(n, chi, p=p, ti=ti)
        
        if check: 
            norm_a = overlap(prmps_a, prmps_a, chi)
            norm_b = overlap(prmps_b, prmps_b, chi)
            if np.abs(1 - norm_a) > 1.e-12:
                print(np.abs(1 - norm_a))
                raise ValueError("State not normalized!")

            if np.abs(1 - norm_b) > 1.e-12:
                print(np.abs(1 - norm_b))
                raise ValueError("State not normalized!")
            
        avg_purity += np.abs(overlap(prmps_a, prmps_b, chi))**(2*k_copies)
    
    avg_purity /= samples
    return avg_purity

def generate_samples(n_list, chi_list, filename, num_samples=5000, ti=False):
    """ generates p=2 scaled samples """
    data = np.zeros((len(n_list), num_samples))
    for i,(n,chi) in tqdm(enumerate(zip(n_list,chi_list))): 
        data[i,:] = np.array(trace_samples(n,chi,num_samples, scaled=True))
        with open(filename, "wb") as f: 
            pickle.dump(data, f)

    return data

def main():
    max_n, n_incr, num_samples = sys.argv[1:]
    n_list = np.arange(2, int(max_n)+1, int(n_incr))
    chi_list_set = [    [4 for n in n_list], 
                        [n for n in n_list],
                        [int(n**1.5) for n in n_list],
                        [n**2 for n in n_list]]
    file_set = ['constant.pickle', 'linear.pickle', 'superlinear.pickle', 'quadratic.pickle']
    
    for (chi_list,filename) in zip(chi_list_set,file_set):
        t1 = time.time()
        generate_samples(n_list, chi_list, filename, int(num_samples))
        t2 = time.time()
        print(f"generated {filename} in {(t2-t1)/60} minutes")
        
if __name__ == "__main__":
    main()