module TensorNetworks
using LinearAlgebra, Combinatorics
using TensorOperations
using LinearMaps
# using DataFrames
using KrylovKit
using DoubleFloats
using ArnoldiMethod
using GenericSchur
using GenericLinearAlgebra #TODO: check if this is necessary
# using Combinatorics
# using SparseArrays
using SparseArrayKit
using Tullio
# using ProgressMeter

# import Distributed.pmap


export TruncationArgs, identityMPS, MPOsite, MPO,
OpenMPS, randomOpenMPS, identityOpenMPS,
LCROpenMPS, randomLCROpenMPS, identityLCROpenMPS,
UMPS, randomUMPS, identityUMPS, transfer_spectrum, boundary, productUMPS,
canonicalize, canonicalize!, iscanonical,
expectation_value, expectation_values, correlator, connected_correlator, matrix_element,
transfer_matrix, transfer_matrices,
prepare_layers, norm, apply_layers,
DMRG, eigenstates,
isingHamBlocks, isingHamGates, IdentityMPO, IsingMPO, HeisenbergMPO,
get_thermal_states, TEBD!, apply_layers_nonunitary, apply_layer_nonunitary!, apply_two_site_gate,
sx, sy, sz, si, s0, ZZ, ZI, IZ, XI, IX,YI,IY, XY, YX, II, Sx, Sy, Sz, XZ,ZX,ZY,YZ,
OrthogonalLinkSite, GenericSite, VirtualSite, LinkSite,
GenericSquareGate, AbstractSquareGate, AbstractGate, Gate,
isleftcanonical, isrightcanonical, data, isunitary,
scalar_product, set_center, set_center!, entanglement_entropy,
entanglement_entropy, IdentityGate, data, compress, qubit,
randomRightOrthogonalSite, randomLeftOrthogonalSite, randomOrthogonalLinkSite, randomGenericSite,
IdentityMPOsite, environment, update_environment!,
ShiftCenter, SubspaceExpand, getindex, setindex!, kron, repeatedgate, *, +, vec,
majorana_coefficients, majorana_measurements

include("types.jl")
include("pauli.jl")
include("mpo.jl")
include("MPSSum.jl")
include("environment.jl")
include("mps.jl")
include("iterative_compression.jl")
include("MPSsite.jl")
include("Gate.jl")
include("AbstractOpenMPS.jl")
include("LCROpenMPS.jl")
include("OpenMPS.jl")
include("UMPS.jl")
# include("CentralUMPS.jl")
include("basic_operations.jl")
include("hamiltonians.jl")
include("coarsegraining.jl")
include("tebd.jl")
include("transfer.jl")
include("dmrg.jl")
include("expectation_values.jl")
include("states.jl")
include("evaluate_wavefunction.jl")
include("majorana.jl")
end # module
