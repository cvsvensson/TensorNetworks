TensorNetworks.jl
================

[![Build status](https://ci.appveyor.com/api/projects/status/1otkx24l3m4bt8m2?svg=true)](https://ci.appveyor.com/project/cvsvensson/tensornetworks)
[![codecov](https://codecov.io/gh/cvsvensson/TensorNetworks/branch/master/graph/badge.svg?token=23H31YT7G9)](https://codecov.io/gh/cvsvensson/TensorNetworks)

Tensor network codes used for research projects. Not intended for public use. The package is not registered nor is it maintained.

Various different types of MPS are supported:
* LCROpenMPS: Finite system which is left-right canonical around some center
* OpenMPS: Canonical ΓΛ representation of Vidal
* UMPS: Uniform MPS. Translationally invariant canonical ΓΛ representation

Algorithms implemented:
* DMRG (for finite systems)
* TEBD (for finite and infinite)

### DMRG
DMRG for the quantum Ising model
```julia
using TensorNetworks
Nchain = 10
Dmax = 20
h,g = (0.226579,0.988821)
mpo = IsingMPO(Nchain, 1, h,g);
mps = canonicalize(randomLCROpenMPS(Nchain, 2, Dmax));
states, energies = eigenstates(mpo, mps, 4; precision = 1e-8, shifter=SubspaceExpand(1.0), maxsweeps = 20);
print(energies) #[-18.98269651078745, -14.996645339327193, -13.221487575678754, -14.99664538769822]
```

Let's compare to eigenvalues from exact diagonalization
```julia
full_hamiltonian = Matrix(mpo); #1024x1024 sparse matrix
using KrylovKit
energiesED, _ = eigsolve(full_hamiltonian,4,:SR);
print(energiesED[1:4]) #[-18.982696525097595, -14.996645398989122, -14.996645398989092, -13.22148761062321]
```

### TEBD
We can find the ground state of the quantum Ising model using TEBD in imaginary time
```julia
ham = isingHamGates(Nchain,1,h,g);
mps_thermal = canonicalize(identityOpenMPS(Nchain, 2, truncation = TruncationArgs(Dmax, 1e-12, true)));
thermal_states, _ = get_thermal_states(mps_thermal, ham, 30, .1, order=2);
expectation_value(thermal_states[1], mpo) #-18.98269652333227 + 0.0im
```

### TEBD for uniform MPS
```julia
ham_gates = isingHamGates(5,1,1,0)[2:3];
mps = canonicalize(identityUMPS(2, 2, truncation = TruncationArgs(Dmax, 1e-12, true)));
infinite_thermal_states, _ = get_thermal_states(mps, ham_gates, 60, .1, order=2);
infinite_ground_state = infinite_thermal_states[1]
energy = (expectation_value(infinite_ground_state,ham_gates[1],1) + expectation_value(infinite_ground_state,ham_gates[2],2))/2;
abs(energy + 4/π) < 1e-4 #close to analytic result
```
