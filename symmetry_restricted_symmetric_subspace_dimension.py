import numpy as np
import scipy as sp
from itertools import permutations
from sympy.combinatorics.permutations import Permutation

def one_copy_projector_ti_subspace(local_dimension, number_sites):
    one_copy_dimension = local_dimension ** number_sites
    bit_strings = []
    for number in range(one_copy_dimension):
        bit_strings.append(format(number, 'b'))
        if len(bit_strings[-1]) <  number_sites:
            bit_strings[-1] = ("0" * (number_sites - len(bit_strings[-1]))) + bit_strings[-1]
    symmetry = np.concatenate([np.arange(number_sites, dtype=int), np.arange(number_sites, dtype=int)])
    permutation_list = [symmetry[i : i + number_sites] for i in range(number_sites)]
    length_permutation_list = len(permutation_list)
    cycle_counts = np.array([Permutation(permutation_list[i]).cycles for i in range(length_permutation_list)], dtype=int)
    return_value = np.sum(local_dimension ** cycle_counts) / len(permutation_list)
    return return_value

def symmetry_symmetric_subspace_dimension(local_dimension, number_sites, k_copies):
    return sp.special.binom(one_copy_projector_ti_subspace(local_dimension, number_sites) + k_copies - 1, k_copies)


square_table_n_k = 10 # number of n and k in the table of symmetry restricted symmetric subspace dimension (inverse purity)
min_number_sites = 2
min_k_copies = 1
local_dimension = 2 # qubits
square_table = np.zeros((square_table_n_k, square_table_n_k))
for n_minus_min in range(square_table_n_k):
    for k_minus_min in range(square_table_n_k):
        square_table[n_minus_min, k_minus_min] = int(symmetry_symmetric_subspace_dimension(local_dimension, n_minus_min + min_number_sites, k_minus_min + min_k_copies)) #assigned to the nth row and kth column

print("For example, for qubits, n=[2, 3, ..., 11] and k=3, the dimensions of the symmtery restricted symmetric subspaces are,")
print(square_table[:,2])