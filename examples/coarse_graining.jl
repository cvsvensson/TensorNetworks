using TensorNetworks, DoubleFloats, LinearAlgebra, TensorOperations, Plots
plotlyjs()
theme(:dark)

##
T = ComplexF64#Complex{Double64}
const d=2
Dmax = 32;
tol = 1e-12;
truncation = TruncationArgs(Dmax,tol,true)
truncationnonormalize = TruncationArgs(Dmax,tol,false)

# Add in the transitional layer of https://arxiv.org/pdf/1602.01166.pdf
transition_gate = Gate(reshape(II + 1im*kron(sx,sx),2,2,2,2)/sqrt(T(2)))
transition_layers = fill(fill(TensorNetworks.auxillerate(transition_gate,transition_gate),2),2);

w, u = TensorNetworks.constr_wu_dim2(T)
@tensor U[:] := u[-1, -3, -5, -7] * u[-2, -4, -6, -8]
#d = size(u, 1)
U = Gate(reshape(U, d^2, d^2, d^2, d^2))
@tensor W[:] := w[-1, -3, -5] * w[-2, -4, -6]
W = reshape(W, d^2, d^2, d^2);

function coarse_grain!(Λs,eigs, mps,cg, U, W)
    # println("norm: ", norm(mps))
    if length(mps) == 1
        mps = TensorNetworks.double(mps)
    end
    println("A: ", sqrt(transfer_spectrum(mps)[1][1]))
    mps = TensorNetworks.apply_layers_nonunitary(mps, [[U]])
    println(sqrt(transfer_spectrum(mps)[1][1]))
    #println(norm.(mps.Λ))
    # println("dnorm: ", norm(mps))
    A = data(mps.Γ[1])
    B = data(GenericSite(mps[2],:right))
    # lA = mps.Λ[1]
    lB = mps.Λ[2]
    # B = B*lA
    @tensor C[:] := B[-1, ul, c] * W[-2, ul, ur] * A[c, ur, -3]
    newmps = UMPS([GenericSite(C,false)], [lB], mps)
    push!(eigs, sqrt(transfer_spectrum(newmps)[1][1]))
    if cg
        newmps.truncation = truncation
    end
    newcanmps = canonicalize(newmps)
    
    push!(Λs, real.(vec(newcanmps.Λ[1])))
    newcanmps.truncation = truncation
    return newcanmps
end
coarse_grain_uw!(Λs,eigs, mps,cg) = coarse_grain!(Λs,eigs, mps,cg,U,W)


##
A0 = A = B = reshape(
    permutedims(
        reshape(kron(Matrix{T}(I, d, d), Matrix{T}(I, d, d)), d, d, d, d),
        [1, 3, 2, 4]),
    d,d^2,d)
lA0 = lA = lB = ones(T,d) / sqrt(T(d))  #(x->x/norm(x))(ComplexF64.(rand(d))) 

mps = UMPS([A0], [lA0], truncation=truncation)
mps2 = canonicalize(TensorNetworks.double(copy(mps)));
t_mps = TensorNetworks.apply_layers(mps2, transition_layers)


##
Λs = []
states = []
eigs = []
mpstemp = canonicalize(copy(t_mps))
for k in 1:10
    mpstemp=coarse_grain_uw!(Λs,eigs, mpstemp,true)
    push!(states,mpstemp)
end