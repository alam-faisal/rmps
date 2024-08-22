import numpy as np
from ncon import *
from tqdm import tqdm
from scipy.special import comb

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

def transfer_matrix(prmps1, prmps2, chi, i): 
    return np.reshape(ncon([prmps1[i], prmps2[i].conj()], [[-1, 1, -3], [-2, 1, -4]]), [chi**2, chi**2])

def transfer_matrices(prmps1, prmps2, chi, ti=False):
    n = len(prmps1)
    if ti: 
        return [transfer_matrix(prmps1, prmps2, chi, 0)]*n
    else: 
        return [transfer_matrix(prmps1, prmps2, chi, i) for i in range(n)]

def overlap(prmps1, prmps2, chi, ti=False):
    if ti: 
        mat = transfer_matrix(prmps1, prmps2, chi, 0)
        evals = np.linalg.eigvals(mat)
        return sum(evals**len(prmps1))
    else: 
        return np.trace(np.linalg.multi_dot(transfer_matrices(prmps1, prmps2, chi, ti=ti)))

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

def haar_purity(n, k_copies, p=2): 
    return 1/comb(p**(n+k_copies)-1, k_copies)