import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, time
sys.path.insert(0,"scripts"); import riemann_gappair as r
w=r.load_unfolded(); rng=np.random.default_rng(7)

def gue_bulk(N, frac=0.5):
    """GUE Hermitian eigenvalues, semicircle-unfolded, middle `frac` (bulk). eigvalsh = fast."""
    A=(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)))/np.sqrt(2)
    ev=np.sort(np.linalg.eigvalsh((A+A.conj().T)/np.sqrt(2)))
    R=2*np.sqrt(N); x=np.clip(ev,-R,R)
    Fc=0.5 + x*np.sqrt(R*R-x*x)/(np.pi*R*R) + np.arcsin(x/R)/np.pi
    wv=N*Fc; lo=int(N*(0.5-frac/2)); hi=int(N*(0.5+frac/2)); return wv[lo:hi]

def corr(seqs):
    cs=[np.corrcoef(np.diff(s)[:-1],np.diff(s)[1:])[0,1] for s in seqs if len(s)>3]
    return float(np.mean(cs)), float(np.std(cs)/np.sqrt(max(len(cs),1)))

print("CUE/GUE consecutive-gap correlation vs block size L -> does it converge to the zeros (-0.357)?",flush=True)
print(f"  {'L':>5} {'#blk':>5} {'zeros(L-blk)':>14} {'GUE(bulk L)':>14} {'diff':>8}",flush=True)
# the zeros are stationary; compare zeros in L-blocks to GUE bulk of matrix size ~2L (bulk frac 0.5 -> ~L levels)
for L in [500,1000,2000,4000,8000]:
    t0=time.time()
    nb=len(w)//L; zb=[w[i*L:(i+1)*L] for i in range(nb)]; cz,ez=corr(zb)
    ng=max(6, min(40, 200000//L))
    gb=[gue_bulk(2*L) for _ in range(ng)]; cg,eg=corr(gb)   # bulk(0.5) of 2L -> ~L levels
    print(f"  {L:>5} {nb:>5} {cz:>7.3f}±{ez:.3f} {cg:>7.3f}±{eg:.3f} {cz-cg:>+8.3f}   ({time.time()-t0:.0f}s, {ng} mats)",flush=True)
