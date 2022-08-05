function maj_ops(n)
    if n==2 
        return [sx,sy]
    elseif n==4 
        return [XI,YI,-ZX,-ZY]
    else
        return [sx,sy]
    end
end
function maj_opsJW(n) 
    if n==2 
        return [1im * sy, -1im * sx]
    elseif n==4 
        return [-1im*YZ, 1im*XZ, 1im * IY, -1im*IX]
    else
        return [1im * sy, -1im * sx]
    end
end
function JWop(n) 
    if n==2 
        return -sz
    elseif n==4 
        return ZZ
    else
        return -sz
    end
end

function antisymmetrise(m::Array{T,N}) where {T,N} 
    Nhalf = Int(N//2)
    perms = permutations(1:Nhalf)
    return mapreduce(p->(-1)^Combinatorics.parity(p)*permutedims(m,vcat(p,p .+ Nhalf)), + ,perms)
end

function close_or_edge_iterator(right_indices, width)
    if length(right_indices)==1|| right_indices[2]-right_indices[1] > width
        return reverse(vcat(1:min(width,right_indices[1]), min(max(width+1, right_indices[1] - width),right_indices[1]):(right_indices[1])))
    else
        return reverse(1:(right_indices[1]))
    end
end
"""
    majorana_measurements(mps, mps2, width)

Return the matrix elements of one- up to four-point functions of majorana operators. 
Works for spinless or spin 1/2 fermions.
"""
function majorana_measurements(mps::TensorNetworks.AbstractMPS{<:TensorNetworks.AbstractSite{T}}, mps2, width::Int) where {T}
    N = length(mps)
    nmaj = size(mps[1],2)
    Tbonds = [TensorNetworks.transfer_matrix_bond((mps,), (mps2,), k, :left) for k in 1:N]
    Tbonds0 = [TensorNetworks.transfer_matrix_bond((mps,),(mps,), k, :left) for k in 1:N]
    JW = JWop(nmaj)
    JWenv = environment((mps,), (MPO(fill(MPOsite(JW), N)), mps2))
    #JWenv0 = environment(mps, MPO(fill(MPOsite(JW), N)))
    env0 = environment((mps,),(mps,))
    env = environment((mps,), (mps2,))
    rightvec = [vec(env.R[k]) for k in 1:N]
    leftvec = [transpose(Tbonds[k] * vec(JWenv.L[k])) for k in 1:N]#vec(Matrix{T}(I,size(mps[k],1),size(mps[k],1))))
    leftvec0 = [transpose(Tbonds0[k] * vec(env0.L[k])) for k in 1:N]
    rightvec0 = [vec(env0.R[k]) for k in 1:N]
    TI = transfer_matrices((mps,), (mps2,), :left)
    TI0 = transfer_matrices((mps,),(mps,), :left)
    TJW = transfer_matrices(mps, MPOsite(JW), mps2, :left)
    TJW0 = transfer_matrices(mps, MPOsite(JW), mps, :left)
    result1::Array{complex(T),2} = zeros(complex(T), N, nmaj)
    result2::Array{real(T),4} = zeros(real(T), N, N, nmaj, nmaj)
    result3::Array{complex(T),6} = zeros(complex(T), N, N, N, nmaj, nmaj, nmaj)
    result4::Array{real(T),8} = zeros(real(T), N, N, N,N, nmaj, nmaj, nmaj,nmaj)

    majops = maj_ops(nmaj)#[TensorNetworks.XI,TensorNetworks.YI,-TensorNetworks.ZX,-TensorNetworks.ZY]
    majopsJW = maj_opsJW(nmaj)#[-1im*TensorNetworks.YZ, 1im*TensorNetworks.XZ, 1im * TensorNetworks.IY, -1im*TensorNetworks.IX]

    T1 = [[transfer_matrix(mps[r], MPOsite(majops[i]), mps2[r], :left) for i in 1:nmaj] for r in 1:N]
    T10 = [[transfer_matrix(mps[r], MPOsite(majops[i]), :left) for i in 1:nmaj] for r in 1:N]
    T1JW = [[transfer_matrix(mps[r], MPOsite(majopsJW[i]), mps2[r], :left) for i in 1:nmaj] for r in 1:N]
    T1JW0 = [[transfer_matrix(mps[r],MPOsite(majopsJW[i]), :left) for i in 1:nmaj] for r in 1:N]
    
    Ttype = typeof(1*T1[1][1])
    
    T2indices::Vector{NTuple{2, Int64}} = vcat([[(k1,k2) for k2 in (1:nmaj)[k1+1:end]] for k1 in 1:nmaj]...)
    T2indicesitr = Base.product(1:nmaj,1:nmaj)
    T2stack(r,i,j) =  (i<j ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majops[j]), mps2[r], :left)
    T2stack0(r,i,j) =  (i<j ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majops[j]), :left)
    T2stackJW(r,i,j) =  (i<j ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majopsJW[j]), mps2[r], :left)
    T2stackJW0(r,i,j) =  (i<j ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majopsJW[j]), :left)
    T2stacked::Vector{Array{Ttype,2}} = [[T2stack(r,i,j) for (i,j) in T2indicesitr] for r in 1:N]
    T2stacked0::Vector{Array{Ttype,2}} = [[T2stack0(r,i,j) for (i,j) in T2indicesitr] for r in 1:N]
    T2stackedJW::Vector{Array{Ttype,2}} = [[T2stackJW(r,i,j) for (i,j) in T2indicesitr] for r in 1:N]
    T2stackedJW0::Vector{Array{Ttype,2}} = [[T2stackJW0(r,i,j) for (i,j) in T2indicesitr] for r in 1:N]

    T3indices::Vector{NTuple{3, Int64}} = [vcat([vcat([[(k1,k2,k3) for k3 in (1:nmaj)[k2+1:end]] for k2 in (1:nmaj)[k1+1:end]]...) for k1 in 1:nmaj]...)...]
    T3stack(r,i,j,k) = (i<j<k ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majops[j]*majops[k]), mps2[r], :left)
    T3stack0(r,i,j,k) = (i<j<k ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majops[j]*majops[k]), :left)
    T3stackJW(r,i,j,k) = (i<j<k ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majops[j]*majopsJW[k]), mps2[r], :left)
    T3stackJW0(r,i,j,k) = (i<j<k ? 1 : 0) * transfer_matrix(mps[r], MPOsite(majops[i]*majops[j]*majopsJW[k]), :left)
    T3indicesitr = Base.product(1:nmaj,1:nmaj,1:nmaj)
    T3stacked::Vector{Array{Ttype,3}} = [[T3stack(r,i,j,k) for (i,j,k) in T3indicesitr] for r in 1:N]
    T3stacked0::Vector{Array{Ttype,3}} = [[T3stack0(r,i,j,k) for (i,j,k) in T3indicesitr] for r in 1:N]
    #T3stackedJW = [[T3stackJW(r,i,j,k) for (i,j,k) in T3indicesitr] for r in 1:N]
    T3stackedJW0::Vector{Array{Ttype,3}} = [[T3stackJW0(r,i,j,k) for (i,j,k) in T3indicesitr] for r in 1:N]

    #T4stacked0 = [transfer_matrix(mps[r], MPOsite(majops[1]*majops[2]*majops[3]*majops[4]), :left) for r in 1:N]
    Threads.@threads for r in 1:N
        R = [op * rightvec0[r] for op in T10[r]]
        if nmaj==4
            T4stacked0 = transfer_matrix(mps[r],MPOsite(majops[1]*majops[2]*majops[3]*majops[4]), mps[r], :left) 
            result4[r,r,r,r, 1,2,3,4] = real(leftvec0[r] * T4stacked0 * rightvec0[r])
        end
        for (i,j) in T2indices
            result2[r, r, i,j] = imag(leftvec0[r] * (T2stacked0[r][i,j]*rightvec0[r]))
        end
        for m in close_or_edge_iterator((r,), width)#:-1:1#max(1, r - width +1)
            if m == r
                M = [op * rightvec0[r] for op in T2stacked0[r]]
            else
                for (i,j,k) in T3indices
                    result4[m, m, m, r, i, j, k,:] .= [real(leftvec0[m] * T3stackedJW0[m][i,j,k] * Rv) for Rv in R]
                end
                M = [op * Rv for (op, Rv) in Base.product(T1JW0[m], R)]
                result2[m, r, :, :] .= [imag(leftvec0[m] * Mv) for Mv in M]
            end
            #M2 = [op * rightvec0[r] for op in T3stacked0[m]]
            for m2 in close_or_edge_iterator((m,r), width)#:-1:1#max(1, r - width +1)
                if m2==m==r
                    M2 = [op * rightvec0[r] for op in T3stacked0[m2]]
                elseif m2==m != r
                    M2 = [op * Rv for (op,Rv) in Base.product(T2stackedJW0[m2],R)]
                else
                    M2 = [op * Mv for (op, Mv) in Base.product(T10[m2], M)]
                    for (i,j) in T2indices
                        result4[m2, m2, m, r, i, j, :,:] .= [real(leftvec0[m2] * T2stacked0[m2][i,j] * Mv) for Mv in M]
                    end
                end
                # end
                for l in close_or_edge_iterator((m2,m,r),width)#m2-1:-1:max(1,r-width+1)
                    if l<m2
                        result4[l, m2, m, r, :, :, :,:] .= [real(leftvec0[l] * op * M2v) for (op, M2v) in Base.product(T1JW0[l], M2)]
                        M2 = map(M2v -> TJW0[l] * M2v, M2)
                    end
                end
                if m2!=m
                    M = map(Mv -> TI0[m2] * Mv, M)
                end
            end
            if m!=r 
                R = map(Rv -> TJW0[m] * Rv, R)
            end
        end
    end
    #Rs = Vector{Vector{Vector{complex(T)}}}(undef,N)
    # Threads.@threads for r in 1:N
    #     R = [op * rightvec[r] for op in T1[r]]
    #     result1[r, :] .= [leftvec[r] * Rv for Rv in R]
    # end
    r2 = antisymmetrise(result2)
    #result3 = three_body_noninteracting(result1, r2)
    Threads.@threads for r in 1:N
        R = [op * rightvec[r] for op in T1[r]]
        result1[r, :] .= [leftvec[r] * Rv for Rv in R]
        for m in r:-1:1#max(1, r - width + 1)
            if m == r
                M = [op * rightvec[m] for op in T2stacked[m]]
                for (i,j,k) in T3indices
                    result3[m, m, m, i, j,k] = 1im*leftvec[m] * T3stacked[m][i,j,k] * rightvec[r]
                end
            else
                M = [op * Rv for (op, Rv) in Base.product(T1JW[m], R)]
                for (i,j) in T2indices
                    result3[m, m, r, i, j,:] .= [1im*leftvec[m] * T2stackedJW[m][i,j] * Rv for Rv in R]
                end
                #result3[m, m, r, 1, 2, :] .= [1im*leftvec[m] * T2stackedJW[m][1,2]*Rv for Rv in R]
            end
            for l in m-1:-1:1#max(1, r - width + 1)
                result3[l, m, r, :, :, :] .= [1im*leftvec[l] * op * Mv for (op, Mv) in Base.product(T1[l], M)]
                M = map(Mv -> TI[l] * Mv, M)
            end
            if m != r
                R = map(Rv -> TJW[m] * Rv, R)
            end
        end
    end

    #r2 = antisymmetrise(result2)
    r3 = antisymmetrise(result3)
    r4 = antisymmetrise(result4)

    return result1, r2, r3,r4#, abmap!,abcmap! #lsola, lsolb, lsolai, lsolbi
end

"""
one_body_noninteracting(mps, mps2, width)

Return the matrix elements of one-point functions of majorana operators. 
Works for spinless or spin 1/2 fermions.
"""
function one_body_noninteracting_majorana_coefficients(mps::TensorNetworks.AbstractMPS{<:TensorNetworks.AbstractSite{T}}, mps2) where {T}
    N = length(mps)
    nmaj = size(mps[1],2)
    Tbonds = [TensorNetworks.transfer_matrix_bond((mps,), (mps2,), k, :left) for k in 1:N]
    JW = JWop(nmaj)
    JWenv = environment((mps,), (MPO(fill(MPOsite(JW), N)), mps2))
    env = environment((mps,), (mps2,))
    rightvec = [vec(env.R[k]) for k in 1:N]
    leftvec = [transpose(Tbonds[k] * vec(JWenv.L[k])) for k in 1:N]#vec(Matrix{T}(I,size(mps[k],1),size(mps[k],1))))
    result1::Array{complex(T),2} = zeros(complex(T), N, nmaj)

    majops = maj_ops(nmaj)#[TensorNetworks.XI,TensorNetworks.YI,-TensorNetworks.ZX,-TensorNetworks.ZY]
    T1 = [[transfer_matrix(mps[r], MPOsite(majops[i]), mps2[r], :left) for i in 1:nmaj] for r in 1:N]

    Threads.@threads for r in 1:N
        R = [op * rightvec[r] for op in T1[r]]
        result1[r, :] .= [leftvec[r] * Rv for Rv in R]
    end
    return result1
end

function three_body_noninteracting(a,r2)
    @tensor b0[:] := a[-1,-4] * r2[-2,-3, -5, -6]
    return antisymmetrise(-b0/2);
end

function majorana_map(rr2::Array{T,4},rr4) where T
    N = size(rr2,1)
    nmaj = size(rr2,3)
    # ag2::Array{T,6} = zeros(T,N,N,N,nmaj,nmaj,nmaj)
    # bg2a::Array{T,2} = zeros(T,N,nmaj)
    # bg4::Array{T,6} = zeros(T,N,N,N,nmaj,nmaj,nmaj)
    function _abmap!(a,b)
        @tensor ag2[:] := a[-1,-4] * rr2[-2,-3, -5, -6]
        @tensor bg2a[:] := b[-1, 1, 2, -2, 3, 4] * rr2[1,2,3,4]
        @tensor bg4[:] := b[-1,1,2,-4,3,4]*rr4[-2,-3,1,2,-5,-6,3,4]
        a .= a - 3*bg2a 
        b .= 3*b + antisymmetrise(-ag2/2 - 3*bg4/2)
        return a,b
    end
    function abmap!(ab)
        a,b = unpack(ab,N,nmaj)
        _abmap!(a,b)
        return vcat(vec(a),vec(b))
    end
    return abmap!
end

function unpack(v,N,nmaj)
    a = reshape(v[1:nmaj*N],N,nmaj)
    b = reshape(v[nmaj*N+1:end],N,N,N,nmaj,nmaj,nmaj)
    return a,b
end

"""
    majorana_coefficients(r1,r2,r3,r4)

Return the one- and three-body coefficients given the measured values.
"""
function majorana_coefficients(r1,r2,r3,r4; tol=1e-12, krylovdim = 10, maxiter = 10)
    N = size(r1,1)
    nmaj = size(r1,2)
    abmap! = majorana_map(r2,r4)
    v = vcat(vec(r1),vec(r3))
    rv = real(v)
    iv = real(-1im*v)
    @time lsolab,info = linsolve(abmap!,rv,rv,maxiter =maxiter, krylovdim=krylovdim, tol=tol);
    println(info)
    @time lsolabi,info = linsolve(abmap!,iv,iv,maxiter =maxiter, krylovdim=krylovdim, tol=tol);
    println(info)
    lsola, lsolb = unpack(lsolab,N,nmaj)
    lsolai, lsolbi = unpack(lsolabi,N,nmaj)
    lsola+1im*lsolai, lsolb+ 1im*lsolbi
end

parityop(N,d) = MPO(fill(MPOsite(JWop(d)), N))
parity(state) = expectation_value(state, parityop(length(state),size(state[1],2)))
function parity_projection(state, pm)
    N = length(state)
    s1 = canonicalize(state)
    #z = 1#evaluate_phase(s1)
    gsp = (TensorNetworks.apply_local_op(s1, Gate(JWop(size(s1[1],2)))) + pm * s1) / sqrt(2)# * z / abs(z))
    gsp2 = canonicalize(canonicalize(TensorNetworks.dense(gsp), center=N), method=:svd)
    return gsp2
end