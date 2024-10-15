Code for computing purities of random MPS with various distributions, boundary conditions and symmetries. 

`tn.py` contains a bare-bones tensor network code built using ncon

`rmps_purity.py` contains methods to generate MPSs with different distributions, boundary conditions and symmetries, as well as the purity of relevant Haar random states. 
The `main()` function generates pairs of random MPS and computes their overlap. This is the part of the code run on a cluster. 
One can also use the `rmps_purity()` function to directly sample purity. 

`process_data.ipynb` is a notebook that reads files containing trace samples and turns them into estimates for the one-norm error between the RMPS ensemble and the Haar ensemble. 
