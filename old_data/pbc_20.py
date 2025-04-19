from rmps_purity import *

n = 20
chi_list = np.array([2,4,6,8,10,12,14,16,18,20])
k_copies = 2
hp = haar_purity(n, k_copies, scaled=False, ti=False)
num_samples = 5000
num_reps = 5 

func = periodic_rmps

means = []
bars = []
for chi in chi_list:
    p_samples = []
    for s in range(num_reps):
        rp = rmps_purity(n, k_copies, chi, samples=num_samples, func=func, quiet=False)
        p_samples.append(rp)
    
    mean, stderr = np.mean(p_samples), np.std(p_samples)
    means.append(mean)
    bars.append(stderr)
    
    with open("p_20_purity.pickle", "wb") as f: 
        pickle.dump((chi_list, means, bars, hp), f)
    
    
