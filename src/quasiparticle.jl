#TODO https://arxiv.org/pdf/1506.01008.pdf


"""
    quasiparticle(mps::UMPS, hamiltonian, length=1)

Return a tensor representing an elementary excitation 
"""
function quasiparticle(mps::UMPS{T}, ham::MPO; nev = 2, all = false) where {T}
    vL, vR = boundary(mps, ham)
    sB = (size(vL, 1), size(ham[1], 3), size(vR, 1))
    function ham_eff(invec)
        tens = reshape(invec, sB)
        @tensoropt (-1, a, c, -3) tens[-1, -2, -3] := vL[-1, l, a] * ham[1].data[l, -2, b, r] * tens[a, b, c] * vR[-3, r, c]
        return vec(tens)
    end
    ham_lin = LinearMap{T}(ham_eff, prod(sB), ishermitian = true)
    if size(ham_lin)[1] < 20 || all
        #print(Matrix(ham_lin))
        println(vec(vL)' * vec(vR))
        evals, evecs = eigen(Matrix(ham_lin))
    else
        evals, evecs = eigsolve(ham_lin, prod(sB), nev, :SR, tol = mps.truncation.tol, ishermitian = true)
        evecs = reduce(hcat, evecs)
    end
    if all
        return evals, evecs
    else
        n = min(length(evals), nev)
        return evals[1:n], [reshape(evecs[:, k], sB) for k in 1:n]
    end
end

function domain_wall() #TODO
    return
end