const DEFAULT_UMPS_DMAX = 20
const DEFAULT_UMPS_TOL = 1e-12
const DEFAULT_UMPS_NORMALIZATION = true
const DEFAULT_UMPS_TRUNCATION = TruncationArgs(DEFAULT_UMPS_DMAX, DEFAULT_UMPS_TOL, DEFAULT_UMPS_NORMALIZATION)
isinfinite(::UMPS) = true
Base.eltype(::Type{UMPS}) = PVSite

function Base.getindex(mps::UMPS, i::Integer)
    i1 = mod1(i, length(mps))
    i2 = mod1(i + 1, length(mps))
    return PVSite(mps.Λ[i1],mps.Γ[i1],mps.Λ[i2])
end
Base.getindex(mps::UMPS, I) = [mps[i] for i in I]

function Base.setindex!(mps::UMPS, v::PVSite, i::Integer)
    i1 = mod1(i, length(mps))
    i2 = mod1(i + 1, length(mps))
    mps.Γ[i1] = v.Γ
    mps.Λ[i1] = v.Λ1
    mps.Λ[i2] = v.Λ2
    return v
end

function UMPS(Γ::Vector{P}, Λ::Vector{V}, mps::UMPS; error = 0) where {P<:AbstractPhysicalSite,V<:AbstractVirtualSite}
    # T = promote_type(eltype(P),eltype(V))
    return UMPS(Γ, Λ, truncation = mps.truncation, error = mps.error + error)
end

# function UMPS(Γ::Vector{Array{T,3}}, mps::UMPS; error = 0) where {T}
#     Λ = LinkSite.([ones(T, size(γ, 1)) / sqrt(size(γ, 1)) for γ in Γ])
#     return UMPS(PhysicalSite.(Γ, ispurification(mps)), Λ, truncation = mps.truncation, error = mps.error + error)
# end

function UMPS(sites::Vector{<:AbstractPVSite}; truncation, error = 0.0)
    Γ, Λ = ΓΛ(sites)
    UMPS(Γ, Λ[1:end-1], truncation = truncation, error = error)
end


"""
	randomUMPS(T::DataType, N, d, D; purification=false, truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)

Return a random UMPS{T}
"""
function randomDenseUMPS(T::DataType, N, d, D; purification = false, truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)
    Γ = [DensePSite{T}(rand(T, D, d, D),purification) for k in 1:N]
    Λ = [LinkSite(ones(T, D)) for k in 1:N]
    mps = UMPS(Γ, Λ, truncation = truncation, error = 0.0)
    return mps
end

"""
	identityUMPS(T::DataType, N, d; truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)

Return the UMPS corresponding to the identity density matrix
"""
function identityUMPS(N, d; T = ComplexF64, truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION)
    # Γ = Vector{Array{T,3}}(undef, N)
    # Λ = Vector{Vector{T}}(undef, N)
    # for i in 1:N
    #     Γ[i] = reshape(Matrix{T}(I, d, d) / sqrt(d), 1, d^2, 1)
    #     Λ[i] = ones(T, 1)
    # end
    Γ = [DensePSite{T}(reshape(Matrix{T}(I, d, d) / sqrt(d), 1, d^2, 1),true) for k in 1:N]
    Λ = [LinkSite(ones(T, 1)) for k in 1:N]
    mps = UMPS(Γ, Λ, truncation = truncation)
    return mps
end
function identityMPS(mps::UMPS{T}) where {T}
    N = length(mps.Γ)
    d = size(mps.Γ[1], 2)
    if ispurification(mps)
        d = Int(sqrt(d))
    end
    trunc = mps.truncation
    return identityUMPS(N, d, T = T, truncation = trunc)
end

function productUMPS(theta, phi)
    Γ = [PhysicalSite(reshape([cos(theta), exp(phi * im) * sin(theta)], (1, 2, 1)),false)]
    Λ = [LinkSite([ComplexF64(1.0)])]
    return UMPS(Γ, Λ)
end

Base.copy(mps::UMPS) = UMPS(copy(mps.Γ),copy(mps.Λ),mps)

# """
# 	reverse_direction(mps::UMPS)

# Flip the spatial direction
# """
# function reverse_direction(mps::UMPS)
# 	Γ = reverse(map(Γ->permutedims(Γ,[3,2,1]),mps.Γ))
# 	Λ = reverse(mps.Λ)
# 	return UMPS(Γ,Λ, mps)
# end

"""
	rotate(mps::UMPS,n::Integer)

Rotate/shift the unit cell `n` steps
"""
function rotate(mps::UMPS, n::Integer)
    return UMPS(circshift(mps.Γ, n), circshift(mps.Λ, n), mps)
end

# %% Transfer
# function transfer_matrix(mps::UMPS, gate::AbstractSquareGate, site::Integer, direction = :left)
# 	oplength = operatorlength(gate)
# 	if ispurification(mps)
# 		gate = auxillerate(gate)
# 	end
#     return transfer_matrix(mps[site:site+oplength-1],op,direction)
# end


"""
	transfer_spectrum(mps::UMPS, direction=:left; nev=1)

Return the spectrum of the transfer matrix of the UMPS
"""
function transfer_spectrum(mps::UMPS{K}, direction::Symbol = :left; nev = 1, kwargs...) where {K}
    T = transfer_matrix((mps,),(mps,), direction)
    D = length(vec(mps.Λ[1]))#(sqrt(size(T,2)))
    nev::Int = min(D^2, nev)
    x0 = vec(Matrix{K}(I, D, D))
    vals, vecs = _eigsolve(T,x0,nev=nev,tol = mps.truncation.tol)
    nev = min(length(vals), nev)
    tensors = [canonicalize_eigenoperator(reshape(vecs[:, k], D, D)) for k in 1:nev]
    return (vals[1:nev]), (tensors) #canonicalize_eigenoperator.(tensors)
end

function transfer_spectrum(mps1::UMPS{K1}, mps2::UMPS{K2}, direction::Symbol = :left; nev = 1) where {K1,K2}
    K = promote_type(K1,K2)
    T = transfer_matrix((mps1,), (mps2,), direction)
    D1 = size(mps1[end], 3)
    D2 = size(mps2[end], 3)
    nev = minimum([D1 * D2, nev])

    x0 = vec(Matrix{eltype(mps1[end])}(I, D1, D2))
    tol = max(mps1.truncation.tol, mps2.truncation.tol)
    vals, vecs = _eigsolve(T,x0,nev=nev,tol = tol)

    nev = min(length(vals), nev)
    tensors = [reshape(vecs[:, k], D1, D2) for k in 1:nev]
    return vals[1:nev], tensors #canonicalize_eigenoperator.(tensors)
end

function LinearAlgebra.norm(mps::UMPS)
    R = boundary(mps, :right)
    L = boundary(mps, :left)
    transpose(L) * transfer_matrix_bond((mps,),(mps,),1,:left)*R
end
"""
	transfer_spectrum(mps::UMPS, mpo::AbstractMPO, direction=:left; nev=1)

Return the spectrum of the transfer matrix of the UMPS, with mpo sandwiched
"""
function transfer_spectrum(mps::UMPS, mpo::AbstractMPO, mps2::UMPS, direction::Symbol = :left; nev = 1)
    T = transfer_matrix((mps,), (mpo, mps2), direction)
    DdD = size(T, 1)
    d = size(mpo[1], 1)
    #N = length(mpo)
    D = Int(sqrt(DdD / d))
    nev = minimum([DdD, nev])
    x0id = Matrix{K}(I, D, D)
    x0v = rand(K, size(mpo[1], 1))
    @tensor x0tens[:] := x0id[-1, -3] * x0v[-2]
    x0 = vec(x0tens)
    tol = max(mps.truncation.tol, mps2.truncation.tol)
    vals, vecs = _eigsolve(T,x0,nev=nev,tol = tol)
    tensors = [reshape(vecs[:, k], D, d, D) for k in 1:nev]
    return vals, tensors #canonicalize_eigenoperator.(tensors)
end

transfer_spectrum((mps1,),(mps2,),dir;nev) = transfer_spectrum(mps1, mps2,dir;nev)

function _eigsolve(transfer::LinearMap{K},x0; nev = 1, tol) where K<:BigNumber
    D = size(transfer,2)
    if D < 50
        vals, vecs = eigen(Matrix(transfer))
        #vals::Vector{K} = vals[end:-1:1]
        #vecs::Matrix{K} = vecs[:, end:-1:1]
    else
        vals, vecs = partialeigen(partialschur(transfer, nev = nev, which = LM(), tol = tol)[1])
    end
    return reverse(vals)::Vector{K} , reverse(vecs,dims=2)::Matrix{K}
end
function _eigsolve(transfer::LinearMap{K},x0; nev = 1, tol) where K
    D = size(transfer,2)
    if D < 50
        vals, vecs = eigen(Matrix(transfer))
        vals::Vector{K} = vals[end:-1:1]
        vecs::Matrix{K} = vecs[:, end:-1:1]
    else
        vals, vecsvec = eigsolve(transfer, x0, nev, :LM,tol=tol)#eigs(T,nev=nev)
        vecs = reduce(hcat, vecsvec)
    end
    return vals, vecs
end


function apply_layers(mps::UMPS, layers)
    sites, err = apply_layers(mps[1:end], layers, mps.truncation, isperiodic = true)
    return UMPS(sites, truncation = mps.truncation, error = mps.error + err)
end
function apply_layers!(mps::UMPS, layers)
    sites, err = apply_layers!(mps[1:end], layers, mps.truncation, isperiodic = true)
    return UMPS(sites, truncation = mps.truncation, error = mps.error + err)
end
function apply_layers_nonunitary(mps::UMPS, layers)
    sites, err = apply_layers_nonunitary(mps[1:end], layers, mps.truncation, isperiodic = true)
    return UMPS(sites, truncation = mps.truncation, error = mps.error + err)
end

# %% Canonicalize
function canonicalize!(mps::UMPS, n)
    for i in 1:n
        apply_identity_layer!(mps, 0)
        apply_identity_layer!(mps, 1)
    end
    return mps
end

# """
# 	canonicalize_cell!(mps::UMPS)

# Make the unit cell canonical
# """
# function canonicalize_cell!(mps::UMPS)
# 	D = length(mps.Λ[1])
# 	N = length(mps.Γ)
# 	Γ = mps.Γ
# 	Λ = mps.Λ

# 	valR, rhoRs = transfer_spectrum(mps,:left,nev=2)
# 	valL, rhoLs = transfer_spectrum(mps,:right,nev=2)
# 	rhoR =  canonicalize_eigenoperator(rhoRs[1])
# 	rhoL =  canonicalize_eigenoperator(rhoLs[1])

#     #Cholesky
# 	if isposdef(rhoR) && isposdef(rhoL)
#     	X = Matrix(cholesky(rhoR, check=true).U)
#     	Y = Matrix(cholesky(rhoL, check=true).U)
# 	else
# 		@warn("Not positive definite. Cholesky failed. Using eigen instead.")
# 		evl, Ul = eigen(Matrix(rhoL))
# 	    evr, Ur = eigen(Matrix(rhoR))
# 	    sevr = sqrt.(complex.(evr))
# 	    sevl = sqrt.(complex.(evl))
# 	    X = Diagonal(sevr)[abs.(sevr) .> mps.truncation.tol,:] * Ur'
# 	    Y = Diagonal(sevl)[abs.(sevl) .> mps.truncation.tol,:] * Ul'
# 	end
# 	F = svd(Y*data(mps.Λ[1])*transpose(X))

#     #U,S,Vt,D,err = truncate_svd(F)

#     #rest
#     YU = VirtualSite(pinv(Y)*F.U ./ (valL[1])^(1/4))
#     VX = VirtualSite(F.Vt*pinv(transpose(X)) ./ (valR[1])^(1/4))
# 	Γ[end] = Γ[end] * YU
# 	Γ[1] = VX*Γ[1]
#     # @tensor Γ[end][:] := data(Γ[end])[-1,-2,3]*YU[3,-3]
#     # @tensor Γ[1][:] := VX[-1,1]*data(Γ[1])[1,-2,-3]
# 	S = LinkSite(F.S)
# 	if mps.truncation.normalize
# 		Λ[1] = S / norm(S)
# 	else
# 		Λ[1] = S
# 	end
# 	return
# end

# function canonicalize(mps::UMPS)
# 	N = length(mps)
# 	if N>2
# 		error("Canonicalize with identity layers if the unit cell is larger than two sites")
# 	end
# 	mps = canonicalize_cell(mps)
# 	if N==2
# 		ΓL, ΓR, error = apply_two_site_gate(mps[1],mps[2], IdentityGate(2), mps.truncation)
# 	    #mps.Γ[1], mps.Λ[2], mps.Γ[2], err = apply_two_site_identity(mps.Γ, mps.Λ[mod1.(1:3,2)], mps.truncation)
# 		#mps.error += err
# 		Γ, Λ = ΓΛ([ΓL, ΓR]) 
# 		mps.Γ = Γ
# 		mps.Λ = Λ[1:end-1]
# 		mps.error += error
# 	end
#     return mps
# end

function canonicalize(mps::UMPS, n)
    for i in 1:n
        mps = apply_identity_layer(mps, 0)
        mps = apply_identity_layer(mps, 1)
    end
    return mps
end


# transfer_matrix_bond(mps::AbstractMPS{_braket(OrthogonalLinkSite)}, site::Integer, dir::Symbol) = (s =Diagonal(data(mps.Λ[site])); kron(s,s))
# transfer_matrix_bond(mps1::AbstractMPS{_braket(OrthogonalLinkSite)}, mps2::AbstractMPS{_braket(OrthogonalLinkSite)}, site::Integer, dir::Symbol) = kron(Diagonal(data(mps1.Λ[site])),Diagonal(data(mps2.Λ[site])))

# """ 
# 	boundary(mps::UMPS, mpo::MPO) 

# Return the left and right dominant eigentensors of the transfer matrix
# """
# function boundary(mps::UMPS, mpo::AbstractMPO) #FIXME should implement https://arxiv.org/pdf/1207.0652.pdf
# 	valR, rhoRs = transfer_spectrum(mps,mpo,:left,nev=2)
# 	valL, rhoLs = transfer_spectrum(mps,mpo,:right,nev=2)
# 	DmpoR = size(mpo[end],4)
# 	DmpoL = size(mpo[1],1)
# 	DR = Int(sqrt(length(rhoRs[:,1])/DmpoR))
# 	DL = Int(sqrt(length(rhoLs[:,1])/DmpoL))
# 	rhoR =  reshape(rhoRs[:,1],DR,DmpoR,DR)
# 	rhoL =  reshape(rhoLs[:,1],DL,DmpoL,DL)
# 	return rhoL, rhoR
# end
# function boundary(mps::UMPS,mpo::AbstractMPO, side::Symbol)
# 	_, rhos = transfer_spectrum(mps,mpo, reverse_direction(side),nev=2)
# 	return canonicalize_eigenoperator(rhos[1])
# end



#TODO Calculate expectation values and effective hamiltonian as in https://arxiv.org/pdf/1207.0652.pdf
#FIXME should implement https://arxiv.org/pdf/1207.0652.pdf
""" 
	effective_hamiltonian(mps::UMPS, mpo::MPO) 

Return the left and right effective_hamiltonian
"""
function effective_hamiltonian(mps::UMPS{T}, mpo::AbstractMPO; direction = :left) where {T}
    Dmpo = size(mpo[end], 1)
    D = length(mps.Λ[1])
    sR = (D, Dmpo, D)
    TL = transfer_matrix((mps,), (mpo,mps), direction)
    TIL = transfer_matrix((mps,),(mps,), direction)
    @warn "Make sure that mpo is lower triangular with identity on the first and last place of the diagonal"
    rhoR = zeros(T, sR) #TODO Sparse array?
    itr = 1:Dmpo
    if direction == :right
        itr = reverse(itr)
    end
    rhoR[:, itr[end], :] = Matrix{T}(I, D, D)
    for k in Dmpo-1:-1:1
        rhoR[:, itr[k], :] = reshape(TL * vec(rhoR), sR)[:, itr[k], :]
    end
    C = rhoR[:, itr[1], :]
    rho = data(mps.Λ[1])^2
    @tensor e0[:] := C[1, 2] * rho[1, 2]
    idvec = vec(Matrix{T}(I, D, D))
    function TI(v)
        v = TIL * v
        return (v - idvec * (vec(rho)' * v))
    end
    linmap = LinearMap{ComplexF64}(TI, D^2)
    hl, info = linsolve(linmap, vec(C) - e0[1] * idvec, 1, -1)
    rhoR[:, itr[1], :] = hl
    return e0[1], rhoR, info
end



"""
	canonicalize_cell(mps::UMPS)

Make the unit cell canonical and return the resulting UMPS
"""
function canonicalize_cell(mps::UMPS{K}) where {K}
    D = length(mps.Λ[1])
    N = length(mps)
    Γcopy = copy(mps.Γ)
    Λcopy = copy(mps.Λ)

    valR, rhoRs = transfer_spectrum(mps, :left, nev = 2)
    valL, rhoLs = transfer_spectrum(mps, :right, nev = 2)
    rhoR = canonicalize_eigenoperator(rhoRs[1])
    rhoL = canonicalize_eigenoperator(rhoLs[1])

    #Cholesky
    if isposdef(rhoR) && isposdef(rhoL)
        X = Matrix(cholesky(rhoR, check = true).U)
        Y = Matrix(cholesky(rhoL, check = true).U)
    else
        @warn "Not positive definite. Cholesky failed. Using eigen instead."
        evl::Vector{K}, Ul::Matrix{K} = eigen(Matrix(rhoL))
        evr::Vector{K}, Ur::Matrix{K} = eigen(Matrix(rhoR))
        sevr = sqrt.(evr)
        sevl = sqrt.(evl)
        X = Diagonal(sevr)[abs.(sevr).>sqrt(mps.truncation.tol), :] * Ur'
        Y = Diagonal(sevl)[abs.(sevl).>sqrt(mps.truncation.tol), :] * Ul'
        #println(minimum(abs.(sevr)))
    end
    m = Y * data(mps.Λ[1]) * transpose(X)
    F = try
        svd(m)
    catch y
        println(y)
        svd(m, alg = LinearAlgebra.QRIteration())
    end
    U, S, Vt, D, err = truncate_svd(F, mps.truncation) #split_truncate!(Y*data(mps.Λ[1])*transpose(X), mps.truncation) 
    Λ = LinkSite(S)
    #rest
    #nf = sqrt(valL[1]/norm(F.S)^2)
    if mps.truncation.normalize
        α = valL[1]^(1 / 4) / sqrt(norm(F.S))
        # println(norm(Λ))
        #β = 1 
    else
        α = one(K)
        #β = 1/norm(mps.Λ[1])
    end
    YU = VirtualSite(pinv(Y) * U / α)
    VX = VirtualSite(Vt * pinv(transpose(X)) / α)
    Γcopy[end] = mps.Γ[end] * YU
    Γcopy[1] = VX * Γcopy[1]
    Λcopy[1] = Λ #/ β
    # @tensor Γcopy[end][:] := Γcopy[end][-1,-2,3]*YU[3,-3]
    # @tensor Γcopy[1][:] := VX[-1,1]*Γcopy[1][1,-2,-3]

    @assert valR[1] ≈ valL[1] "Left and right eigenvalues not equal: $(valR) !≈ $(valL)"
    return UMPS(Γcopy, Λcopy, mps, error = err)
end

function canonicalize(mps::UMPS)
    N = length(mps.Γ)
    #Γ = similar(mps.Γ)
    #Λ = deepcopy(mps.Λ)
    if N > 2
        error("Canonicalize with identity layers if the unit cell is larger than two sites")
    end
    mps = canonicalize_cell(mps)
    if N == 1
        mpsout = mps
    elseif N == 2
        #Γ[1],Λ2,Γ[2], err = apply_two_site_identity(mps.Γ, mps.Λ[mod1.(1:3,2)], mps.truncation)
        # println(mps[1].Γ)
        # println(mps[1].Λ2)
        # println(mps[2].Γ)
        # println(mps[2].Λ2)
        # println(transfer_spectrum(mps)[2])
        ΓL, ΓR, err = apply_two_site_gate(mps[1], mps[2], IdentityGate(Val(2)), mps.truncation)
        mpsout = UMPS([ΓL, ΓR], truncation = mps.truncation, error = err)
    else
        error("Canonicalizing $N unit sites not implemented")
        return mps
    end
    return mpsout
end

function apply_mpo(mps::UMPS, mpo)
    Nmpo = length(mpo)
    Nmps = length(mps.Γ)
    N = Int(Nmps * Nmpo / gcd(Nmps, Nmpo))
    mpo = mpo[mod1.(1:N, Nmpo)]
    Γ = mps.Γ[mod1.(1:N, Nmps)]
    Λ = mps.Λ[mod1.(1:N, Nmps)]
    Γout = similar(Γ)
    Λout = similar(Λ)
    for i in 1:N
        @tensor tens[:] := Γ[i][-1, c, -4] * mpo[i][-2, -3, c, -5]
        st = size(tens)
        Γout[i] = reshape(tens, st[1] * st[2], st[3], st[4] * st[5])
        @tensor Λtemp[:] := Λ[i][-1] * ones(st[2])[-2] #FIXME Λ is now a diagonal matrix
        Λout[i] = reshape(Λtemp, st[1] * st[2])
    end
    return UMPS(Γout, Λout, mps)
end

# %% TEBD
"""
	double(mps::UMPS)

Return an UMPS with double the size of the unit cell
"""
function double(mps::UMPS)
    N = length(mps.Γ)
    return UMPS(vcat(mps.Γ,copy.(mps.Γ)), vcat(mps.Λ,copy.(mps.Λ)), mps)
end



#%% Expectation values
function expectation_value(mps::UMPS, op::Array{T_op,N_op}, site::Integer) where {T_op<:Number,N_op}
    opLength = operatorlength(op)
    N = length(mps.Γ)
    if ispurification(mps)
        op = auxillerate(op)
    end
    if opLength == 1
        val = expectation_value_one_site(mps.Λ[site], mps.Γ[site], mps.Λ[mod1(site + 1, N)], op)
    elseif opLength == 2
        val = expectation_value_two_site(mps.Γ[mod1.(site:site+1, N)], mps.Λ[mod1.(site:site+2, N)], op)
    else
        error("Expectation value not implemented for operators of this size")
    end
    return val
end

#TODO check if this function works, especially for different sizes of operators
function correlator(mps::UMPS{T}, op1, op2, n) where {T}
    opsize = size(op1)
    oplength = operatorlength(op1)
    transfers = transfer_matrices(mps, :right)
    N = length(transfers)
    Ts = transfers[mod1.((1+oplength):(n+oplength-1), N)]
    Rs = Vector{Transpose{T,Vector{T}}}(undef, N)
    for k in 1:N
        Tfinal = transfer_matrix(mps, op2, k, :left)
        st = Int(sqrt(size(Tfinal, 2)))
        R = reshape(Matrix{T}(I, st, st), st^2)
        Rs[k] = transpose(Tfinal * R)
    end
    Tstart = transfer_matrix(mps, op1, 1, :right)
    sl = Int(sqrt(size(Tstart, 2)))
    L = reshape(Matrix(I, sl, sl), sl^2)
    L = Tstart * L
    vals = Array{ComplexF64,1}(undef, n - 1)
    for k = 1:n-1
        Λ = Diagonal(mps.Λ[mod1.(k + oplength, N)])
        middle = kron(Λ, Λ)
        vals[k] = Rs[mod1(k + oplength, N)] * middle * L
        L = Ts[k] * L
    end
    return vals
end

"""
     canonicalize_eigenoperator(rho)

make the dominant eigenvector hermitian
"""
function canonicalize_eigenoperator(rho::AbstractMatrix)
    rhoH = Hermitian((rho + rho') / 2)
    return rhoH / (sign(tr(rhoH))*norm(rhoH)) * sqrt(size(rhoH, 1))
end


# %% Entropy

function renyi(mps;tol=1e-6)
    T = transfer_matrix_squared(mps,:right)
    vals,vecs, info = eigsolve(T,size(T,2),1,tol=tol,maxiter=6,krylovdim=8)#,1)eigsolve(heff, vec(x0), nev, :SR, tol = prec, ishermitian = true, maxiter = 3, krylovdim = 20)
	println(info)
    return -log2(vals[1])
end
function transfer_matrix_squared(mps,dir=:left)
    Ts = transfer_matrices_squared(mps, dir)
    if dir ==:right
        return prod(reverse(Ts))
    else
        return prod(Ts)
    end
end
function transfer_matrices_squared(mps,dir)
    [_local_transfer_matrix_squared(site,dir) for site in mps]
end
function _local_transfer_matrix_squared(site,dir)
    if dir==:right
        return _transfer_matrix_right_squared(data(site, reverse_direction(dir)))
    elseif dir==:left
        return _transfer_matrix_right_squared(reverse_direction(data(site, reverse_direction(dir))))
    else
        @error "Choose direction :left or :right"
    end
end
function _transfer_matrix_right_squared(s)
    ss=size(s)
    A = reshape(s,ss[1],Int(sqrt(ss[2])),Int(sqrt(ss[2])),ss[3])
    sA=size(A)
    function contract(R)
        temp = reshape(R,sA[1],sA[1],sA[1],sA[1])
        @tensor out[:] := A[r1,u,c1,-1]*(conj(A[r2,c2,c1,-2])*(A[r3,c2,c3,-3]*(conj(A[r4,u,c3,-4])*temp[r1,r2,r3,r4])))
        return vec(out)
    end
    T = LinearMap{ComplexF64}(contract,sA[4]^4,sA[1]^4)
    return T
end

# %%
function saveUMPS(mps, filename)
    jldopen(filename, "w") do file
        writeOpenMPS(file, mps)
    end
end

function writeUMPS(parent, mps)
    write(parent, "Gamma", mps.Γ)
    write(parent, "Lambda", mps.Λ)
    write(parent, "Purification", ispurification(mps))
    write(parent, "Dmax", mps.truncation.Dmax)
    write(parent, "tol", mps.truncation.tol)
    write(parent, "normalize", mps.truncation.normalize)
    write(parent, "error", mps.error)
end

function readUMPS(io)
    Γ = read(io, "Gamma")
    Λ = read(io, "Lambda")
    purification = read(io, "Purification")
    Dmax = read(io, "Dmax")
    tol = read(io, "tol")
    normalize = read(io, "normalize")
    error = read(io, "error")
    trunc = TruncationArgs(Dmax, tol, normalize)
    mps = UMPS(Γ, Λ, purification = purification, truncation = trunc, error = error)
    return mps
end

function loadUMPS(filename)
    jldopen(filename, "r") do file
        global mps
        mps = readOpenMPS(file)
    end
    return mps
end
