"""
    evaluate_wavefunction(mps, indices)

Return the value of the state represented by `mps` at the given `indices`. 
The latter is a vector of the same length as the mps.
"""
function evaluate_wavefunction(mps, indices)
    N = length(mps)
    @assert length(indices) == N
    sizes = size.(mps, 2)
    @assert all(indices .<= sizes)
    sites = Vector{eltype(mps)}(undef, N)
    T = numtype(mps)
    for k in 1:N
        temp_tensor = zeros(T,1, sizes[k], 1)
        temp_tensor[indices[k]] = one(T)
        sites[k] = PhysicalSite(temp_tensor, false)
    end
    scalar_product(OpenPMPS(sites), mps)
end