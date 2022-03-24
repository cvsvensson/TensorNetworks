using TensorNetworks, TensorOperations, SnoopCompile
#TensorOperations.disable_cache()
#disable_blas()
Ts = subtypes(Any);
N = 5
d = 2 
Dmax = 50
prec = 1e-14
truncation = TruncationArgs(Dmax, prec, true)
J = 1.0
h = .5
g = 0.001
hammpo = IsingMPO(N,J,h,g);
hamgates = isingHamGates(N,J,h,g);
hammpo = HeisenbergS1MPO(N,1.0,1,1,0);
# %% DMRG
n_states= 2
# initialmps=randomLCROpenMPS(N,d, Dmax; truncation = truncation);
tinf = @snoopi_deep begin
    initialmps=randomLCROpenMPS(N,d, Dmax; truncation = truncation);
    eigenstates(hammpo, initialmps, n_states, precision = prec);
    end
using Profile  
@profile begin
    initialmps=randomLCROpenMPS(N,d, Dmax; truncation = truncation);
    eigenstates(hammpo, initialmps, n_states, precision = prec);
    end 
import PyPlot
mref, ax = pgdsgui(tinf);

collect_for(mref[], tinf)

itrigs = inference_triggers(tinf)
mtrigs = accumulate_by_source(Method, itrigs)
modtrigs = filtermod(TensorNetworks, mtrigs)

summary.(modtrigs)