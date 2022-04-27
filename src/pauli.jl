
const sx = [0 1; 1 0]
const sy = [0 -1im; 1im 0]
const sz = [1 0; 0 -1]
const si = [1 0; 0 1]
const s0 = [0 0; 0 0]
const splus = [0 1; 0 0]
const sminus = [0 0; 1 0]
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
const SpI = kron(si,splus)
const SpX = kron(sx,splus)
const SpY = kron(sy,splus)
const SpZ = kron(sz,splus)
const ISp = kron(splus,si)
const XSp = kron(splus,sx)
const YSp = kron(splus,sy)
const ZSp = kron(splus,sz)
const SmI = kron(si,sminus)
const SmX = kron(sx,sminus)
const SmY = kron(sy,sminus)
const SmZ = kron(sz,sminus)
const ISm = kron(sminus,si)
const XSm = kron(sminus,sx)
const YSm = kron(sminus,sy)
const ZSm = kron(sminus,sz)
const SmSm = kron(sminus,sminus)
const SmSp = kron(splus,sminus)
const SpSp = kron(splus,splus)
const SpSm = kron(sminus,splus)


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