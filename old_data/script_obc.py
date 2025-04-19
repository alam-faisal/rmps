from rmps_purity import *

n = 10
chi_list = np.array([2,4,8,10,12,16])
k_copies = 2
hp = haar_purity(n, k_copies, scaled=False, ti=False)
num_samples = 5000
num_reps = 5 

func = open_rmps

data = []
for chi in chi_list:
    p_samples = []
    for s in range(num_reps):
        rp = rmps_purity(n, k_copies, chi, samples=num_samples, func=func, quiet=False)
        p_samples.append(rp)
    
    data.append(p_samples)
    
means = []
bars = []
for p_samples in data: 
    mean, stderr = np.mean(p_samples), np.stderr(p_samples)
    means.append(mean)
    bars.append(stderr)
    
with open("o_10_purity.pickle", "wb") as f: 
    pickle.dump((chi_list, means, bars, hp), f)