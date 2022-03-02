"""
    evaluate_wavefunction(mps, indices)

Return the value of the state represented by `mps` at the given `indices`. 
The latter is a vector of the same length as the mps.
"""
function evaluate_wavefunction(mps, indices)
    N = length(mps)
    @assert length(indices) == N
    sizes = size.(mps,2)
    @assert  all(indices .<= sizes)
    sites = Vector{GenericSite{ComplexF64}}(undef,N)
    for k in 1:N
        temp_tensor = zeros(1,sizes[k],1)
        temp_tensor[indices[k]] = 1
        sites[k] = GenericSite{ComplexF64}(temp_tensor,false)
    end
    scalar_product(LCROpenMPS(sites),mps)
end

function test_ew()
    N = 5
    qubits = [qubit(rand(),rand()) for k in 1:N]
    mps = LCROpenMPS(qubits)
    indices = rand([1,2], N)
    println(indices)
    wf = evaluate_wavefunction(mps, indices)
    println(wf)
    println(prod([q[1,indices[k],1] for (k,q) in enumerate(qubits)]))
    println(wf ≈ prod([q[1,indices[k],1] for (k,q) in enumerate(qubits)]))

    # Test for entangled states and larger physical dim
    d=5
    sites = [randomLeftOrthogonalSite(1,d,d) , randomLeftOrthogonalSite(d,d,1)]
    mps = LCROpenMPS(sites)
    for (n1,n2) in Iterators.product(1:d,1:d)
        wf = evaluate_wavefunction(mps, [n1,n2])
        println(wf ≈ transpose(sites[1][1,n1,:]) * sites[2][:,n2,1])
    end
end