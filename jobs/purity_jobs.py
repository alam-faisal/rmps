import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmps_purity import *
from tqdm import tqdm
import pickle

if len(sys.argv) < 4 or len(sys.argv) > 6:
    print("Usage: python script.py mps_type num_sites repeats [distrib] [params]")
    print("Example: python script.py 'open staggered' 10 15")
    print("Example with optional args: python script.py 'open staggered' 10 15 'gaussian' 0.5")
    sys.exit(1)

mps_type = sys.argv[1]
num_sites = int(sys.argv[2])
repeats = int(sys.argv[3])

# Handle optional arguments
distrib = sys.argv[4] if len(sys.argv) > 4 else None
param = float(sys.argv[5]) if len(sys.argv) > 5 else None

# Select MPS function based on type
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

# Calculate purities for different chi values
chi_values = list(range(2, 20, 2))
k_copies = 2
purity_avgs = []
purity_stds = []

# Build keyword arguments for rmps_purity
kwargs = {'samples': 5000, 'func': func, 'quiet': True}
if distrib is not None:
    kwargs['distrib'] = distrib
if param is not None:
    kwargs['param'] = param

for chi in tqdm(chi_values):
    purities = [
        rmps_purity(num_sites, k_copies, chi, **kwargs)
        for _ in range(repeats)
    ]
    purity_avgs.append(np.mean(purities))
    purity_stds.append(np.std(purities))

# Modify filename to include distribution information if provided
filename_parts = [f"rmps_{mps_type.replace(' ', '_')}_{num_sites}"]
if distrib is not None:
    filename_parts.append(f"{distrib}")
    if param is not None:
        filename_parts.append(f"{param}")
filename = "_".join(filename_parts) + ".pickle"

with open(filename, 'wb') as f:
    pickle.dump((chi_values, purity_avgs, purity_stds), f)

print(f"Results saved to {filename}")
