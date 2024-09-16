from ncon import *
import numpy as np
import copy
from scipy.linalg import svd

class TensorNetwork:
    def __init__(self, sites):
        self.sites = sites
        self.num_sites = len(sites)
        self.local_dim = sites[0].shape[-1]

    def __matmul__(self, other):
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_skeleton(self): 
        return [site.shape for site in self.sites]
    
    def get_max_dim(self):
        return max(max(shape) for shape in self.get_skeleton())
    
    def get_site_norms(self): 
        return [np.linalg.norm(np.ravel(site)) for site in self.sites]
    
    def conj(self): 
        sites = [copy.deepcopy(site).conj() for site in self.sites]
        return type(self)(sites)
        
class MPS(TensorNetwork): 
    """ a site is a np.array of shape (left_bond_dim, right_bond_dim, local_dim) """
    def __init__(self, sites):
        super().__init__(sites)
    
    def __matmul__(self, other):
        if isinstance(other, MPS):
            return self.overlap(other)
        else:
            raise TypeError("Unsupported operand type for @")
    
    def overlap(self, other, scaled=False):
        scale = np.sqrt(self.local_dim) if scaled else 1.0
        
        tensor = ncon((self.sites[0], other.sites[0].conj()), ([-1,-3,1],[-2,-4,1])) * scale
        for i in range(1,self.num_sites):
            tensor = ncon((tensor, self.sites[i]), ([-1,-2,1,-3],[1,-4,-5]))
            tensor = ncon((tensor, other.sites[i].conj()), ([-1,-2,1,-3,2],[1,-4,2])) * scale
        return np.trace(tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2] * tensor.shape[3]))

    def norm(self): 
        return self.overlap(self)
    
    def normalize(self): 
        n = self.norm()
        self.sites = [site/np.sqrt(np.abs(n))**(1/self.num_sites) for site in self.sites]
    
class MPO(TensorNetwork): 
    """ a site is a np.array of shape (left_bond_dim, right_bond_dim, top_local_dim, bottom_local_dim) """
    def __init__(self, sites):
        super().__init__(sites)
    
    def __matmul__(self, other):
        if isinstance(other, MPS):
            return self.act(other)
        elif isinstance(other, MPO):
            return self.compose(other)
        else:
            raise TypeError("Unsupported operand type for @")
            
    def act(self, mps):
        new_sites = []
        for site_a, site_b in zip(self.sites, mps.sites):
            ldim = site_a.shape[0] * site_b.shape[0]
            rdim = site_a.shape[1] * site_b.shape[1]
            new_site = ncon((site_a, site_b), ([-1,-3,-5,1],[-2,-4,1])).reshape(ldim, rdim, mps.local_dim)
            new_sites.append(new_site)
        return MPS(new_sites)
    
    def compose(self, mpo):
        new_sites = []
        for site_a, site_b in zip(self.sites, mpo.sites):
            ldim = site_a.shape[0] * site_b.shape[0]
            rdim = site_a.shape[1] * site_b.shape[1]
            sdim = self.local_dim
            new_site = ncon((site_a, site_b), ([-1,-3,-5,1],[-2,-4,1,-6])).reshape(ldim, rdim, sdim, sdim)
            new_sites.append(new_site)
        return MPO(new_sites)
    
    def overlap(self, other): 
        tensor = ncon((self.sites[0], other.sites[0]), ([-1,-3,1,2],[-2,-4,2,1]))/self.local_dim
        for i in range(1,self.num_sites):
            tensor = ncon((tensor, self.sites[i]), ([-1,-2,1,-3],[1,-4,-5,-6]))
            tensor = ncon((tensor, other.sites[i]), ([-1,-2,1,-3,2,3],[1,-4,3,2]))/self.local_dim
        return np.trace(tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2] * tensor.shape[3]))
    
    def trace(self): 
        tr = ncon((self.sites[0]), ([-1,-2,1,1]))/self.local_dim
        for i in range(1,self.num_sites): 
            tr = ncon((tr, self.sites[i]), ([-1,1],[1,-2,2,2]))/self.local_dim
        return np.trace(tr)
    
    def to_matrix(self): 
        tensor = self.sites[0]
        for i in range(1,self.num_sites):
            left_dim = tensor.shape[0]
            right_dim = self.sites[i].shape[1]
            local_dim = self.local_dim**(i+1)
            tensor = ncon((tensor, self.sites[i]), ([-1,1,-3,-5],[1,-2,-4,-6])).reshape(left_dim,right_dim,local_dim,local_dim)
        return ncon((tensor), (1,1,-1,-2))
    
    def diag(self): 
        sites = copy.deepcopy(self.sites)
        for k in range(self.num_sites):
            site = sites[k]
            s1,s2 = site.shape[0], site.shape[1]
            for i in range(s1):
                for j in range(s2):
                    site[i,j,:,:] = np.diag(np.diag(site[i,j,:,:]))
            sites[k] = site
        return MPO(sites)

###################################
##### Tensor methods ##############
###################################

def decompose_site(data, min_sv_ratio=None, max_dim=None): 
    """ 
    decomposes sites of an MPS or MPDO after action of a two-qubit gate 
    data.shape = (top_qubit_top_spin, top_qubit_bottom_spin, bottom_qubit_top_spin, 
                    bottom_qubit_bottom_spin, left_bond, right_bond) 
    """
    sh = data.shape
    is_mpo = len(sh) == 6
    
    if is_mpo:
        data = data.transpose(0,1,4,2,3,5).reshape(sh[0]*sh[1]*sh[4], sh[2]*sh[3]*sh[5])
    else:
        data = data.transpose(0,2,1,3).reshape(sh[0]*sh[2], sh[1]*sh[3])
    
    u,s,vh = svd(data, full_matrices=False, lapack_driver='gesvd')
    
    if min_sv_ratio is not None: 
        s = s[s>min_sv_ratio*s[0]]
    elif max_dim is not None:
        dim = min(max_dim, len(s[s>default_min_sv_ratio*s[0]]))
        s = s[:dim]
    
    u = u[:,:len(s)] @ np.diag(np.sqrt(s))
    vh = np.diag(np.sqrt(s)) @ vh[:len(s),:]

    if is_mpo:
        top_site = u.reshape(sh[0],sh[1],sh[4],len(s)).transpose(2,3,0,1)
        bottom_site = vh.reshape(len(s),sh[2],sh[3],sh[5]).transpose(0,3,1,2)
    else:
        top_site = u.reshape(sh[0],sh[2],len(s)).transpose(1,2,0)
        bottom_site = vh.reshape(len(s),sh[1],sh[3]).transpose(0,2,1)
        
    return top_site, bottom_site

#################################
### Special initializations #####
#################################

def all_zero_mpdo(n, local_dim=2):
    mat = np.zeros((local_dim, local_dim))
    mat[0,0] = 1.0
    return MPO([mat[np.newaxis,np.newaxis,:,:] for i in range(n)])

def rand_mpo(n, chi, boundary='open', local_dim=2):
    if boundary == 'open': 
        sites = [np.random.rand(1,chi,local_dim,local_dim)] + [np.random.rand(chi,chi,local_dim,local_dim)
                     for _ in range(n-2)] + [np.random.rand(chi,1,local_dim,local_dim)]
    elif boundary == 'periodic': 
        sites = [np.random.rand(chi,chi,local_dim,local_dim) for _ in range(n)]
    else: 
        raise ValueError(f"Invalid boundary: {boundary}")
        
    return MPO(sites)