
const sx = [0 1; 1 0]
const sy = [0 -1im; 1im 0]
const sz = [1 0; 0 -1]
const si = [1 0; 0 1]
const s0 = [0 0; 0 0]
const ZZ = kron(sz, sz)
const YY = kron(sy, sy)
const XX = kron(sx, sx)
const ZI = kron(si, sz)
const IZ = kron(sz, si)
const XI = kron(si, sx)
const IX = kron(sx, si)
const XY = kron(sy, sx)
const YX = kron(sx, sy)
const II = kron(si, si)

# const sx1 = [0 1 0; 1 0 1; 0 1 0]/sqrt(2)
# const sy1 = 1im*[0 -1 0; 1 0 -1; 0 1 0]/sqrt(2)
# const sz1 = [1 0 0; 0 0 0; 0 0 -1]

function Sx(s)
    @assert isinteger(2s)
    s2 = Int(2s)
    upper = [sqrt((s + 1) * 2k - k * (k + 1)) / 2 for k in 1:s2]
    middle = zeros(s2 + 1)
    return Tridiagonal(upper, middle, upper)
end
function Sy(s)
    @assert isinteger(2s)
    s2 = Int(2s)
    upper = [-1im * sqrt((s + 1) * 2k - k * (k + 1)) / 2 for k in 1:s2]
    middle = zeros(eltype(upper), s2 + 1)
    return Tridiagonal(-upper, middle, upper)
end
function Sz(s)
    @assert isinteger(2s)
    s2 = Int(2s)
    middle = [s + 1 - k for k in 1:s2+1]
    return Diagonal(middle)
end
function Si(s)
    Matrix(I, Int(2s + 1), Int(2s + 1))
end