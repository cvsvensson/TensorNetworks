Base.IndexStyle(::Type{<:AbstractMPS}) = IndexLinear()
Base.size(mps::AbstractMPS) = size(sites(mps))

ispurification(mps::AbstractMPS) = ispurification(mps[1])

Base.show(io::IO, mps::AbstractMPS) =
    print(io, "MPS: ", typeof(mps), "\nSites: ", eltype(mps) ,"\nLength: ", length(mps), "\nTruncation: ", mps.truncation)
Base.show(io::IO, m::MIME"text/plain", mps::AbstractMPS) = show(io,mps)

function scalar_product(mps1::AbstractMPS, mps2::AbstractMPS)
	K = numtype(mps1,mps2)
	Ts::Vector{LinearMap{K}} = transfer_matrices(mps1,mps2)
	vl = transfer_matrix_bond(mps1,mps2,1,:right)*boundaryvec(mps1,mps2,:right)
	vr::Vector{K} = boundaryvec(mps1,mps2,:right)
	for k in length(mps1):-1:1
		vr = Ts[k] * vr
	end
	return transpose(vr)*vl
end

LinearAlgebra.norm(mps::AbstractMPS) = scalar_product(mps,mps)

function prepare_layers(mps::AbstractMPS, gs::Vector{<:AbstractSquareGate}, dt, trotter_order)
	gates = ispurification(mps) ? auxillerate.(gs) : gs
	return prepare_layers(gates,dt,trotter_order)
end

"""
	get_thermal_states(mps, hamGates, betas, dbeta, order=2)

Return a list of thermal states with the specified betas
"""
function get_thermal_states(mps::AbstractMPS, hamGates, βs, dβ; order=2)
	Nβ = length(βs)
	mps = identityMPS(mps)
	canonicalize!(mps)
	mpss = Array{typeof(mps),1}(undef,Nβ)
	layers = prepare_layers(mps, hamGates,dβ*1im/2, order)
	β=0
	βout = Float64[]
	for n in 1:Nβ
		Nsteps = floor((βs[n]-β)/dβ)
		count=0
		while β < βs[n]
			mps = apply_layers_nonunitary(mps,layers)
			canonicalize!(mps)
			β += dβ
			count+=1
			if mod(count,floor(Nsteps/10))==0
				print("-",string(count/Nsteps)[1:3],"-")
			end
		end
		println(": State ", n ,"/",Nβ, " done.")
		push!(βout,β)
		mpss[n] = copy(mps)
	end
	return mpss, βout
end

function imaginaryTEBD(mps::AbstractMPS, hamGates, βtotal, dβ; order=2)
	mps = deepcopy(mps)
	canonicalize!(mps)
	layers = prepare_layers(mps, hamGates,dβ*1im, order)
	β=0
	count=0
	Nsteps = βtotal/dβ
	while β < βtotal
		mps = apply_layers_nonunitary(mps,layers)
		canonicalize!(mps)
		β += dβ
		count+=1
		if mod(count,floor(Nsteps/10))==0
			print("-",string(count/Nsteps)[1:3],"-")
		end
	end
	return mps
end


"""
	apply_local_op!(mps,op)

Apply the operator at every site
"""
function apply_local_op(mps,op)
    N = length(mps.Γ)
	Γ = similar(mps.Γ)
	if ispurification(mps)
		op = auxillerate(op)
	end
 	for n in 1:N 
    	@tensor Γ[n][:] = mps.Γ[n][-1,2,-3]*op[-2,2] #TODO Replace by Tullio or matrix mult?
	end
end

"""
	apply_local_op!(mps,op, site)

Apply the operator at the site
"""
function apply_local_op!(mps,op,site::Integer) #FIXME pass through to operation on site
    #N = length(mps.Γ)
	if ispurification(mps)
		op = auxillerate(op)
	end
    @tensor mps.Γ[site][:] := mps.Γ[site][-1,2,-3]*op[-2,2] #TODO Replace by Tullio or matrix mult?
end

# function Base.:+(mps1::AbstractMPS, mps2::AbstractMPS)
# 	@assert length(mps1) == length(mps2) "mps1 and mps2 have different lengths: $(length(mps1)) vs $(length(mps2))"
# 	N = length(mps1)
# 	mps3 = similar(mps1)
# 	for k in 1:N
# 		mps3[k] = mps1[k] + mps2[k]
# 	end
# 	if boundaryconditions(mps1) == OpenBoundary
# 		mps3[1]
# end

# function Base.:+(site1::GenericSite, site2::GenericSite)
# 	s1 = size(site1)
# 	s2 = size(site2)
# 	d = s1[2]
# 	@assert d == s2[2] "Sites have different physical dimension, can't add them. $d vs $(s2[2])"
# 	T = promote_type(eltype.([site1,site2])...)
# 	largesite = zeros(T,s1[1]+s2[1], d, s1[3]+s2[3])
# 	largesite[1:s1[1],1:d,1:s1[3]] = data(site1)
# 	largesite[s1[1]+1:s1[1]+s2[1],1:d,s1[3]+1:s1[3]+s2[3]] = data(site2)
# 	return GenericSite(largesite, ispurification(site1))
# end


# function Base.:+(site1::OrthogonalLinkSite, site2::OrthogonalLinkSite)
# 	Λ1 = LinkSite(vcat(diag(data(link(site1,:left))), diag(data(link(site2,:left)))))
# 	Λ2 = LinkSite(vcat(diag(data(link(site1,:right))), diag(data(link(site2,:right)))))
# 	Γ = site1.Γ + site2.Γ
# 	return OrthogonalLinkSite(Λ1,Γ,Λ2,check=false)
# end
# Base.kron(l1::LinkSite,l2::LinkSite) = LinkSite(kron(data(l1),data(l2)))




# function Base.:*(x::Number, site::OrthogonalLinkSite)
# 	OrthogonalLinkSite(link(site,:left), x*site.Γ, link(site,:right), check=false)
# end