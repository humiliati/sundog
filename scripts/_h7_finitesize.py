import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys
sys.path.insert(0,"scripts"); import riemann_gappair as r
from scipy.stats import unitary_group
w=r.load_unfolded(); rng=np.random.default_rng(7)
def corr_blocks(blocks):
    cs=[np.corrcoef(np.diff(b)[:-1],np.diff(b)[1:])[0,1] for b in blocks if len(b)>3]
    return float(np.mean(cs)), float(np.std(cs)/np.sqrt(len(cs)))
print("does CUE consecutive-gap correlation CONVERGE to the zeros as block size L grows?")
print(f"  {'L':>5} {'zeros':>16} {'CUE':>16} {'diff':>8}")
for L in [250,500,1000,2000,4000]:
    nb=len(w)//L; zb=[w[i*L:(i+1)*L] for i in range(nb)]
    cz,ez=corr_blocks(zb)
    ncue=min(nb,30)
    cue=[np.sort(np.angle(np.linalg.eigvals(unitary_group.rvs(L,random_state=rng)))%(2*np.pi))*L/(2*np.pi) for _ in range(ncue)]
    cc,ec=corr_blocks(cue)
    print(f"  {L:>5} {cz:>8.3f}±{ez:.3f} {cc:>8.3f}±{ec:.3f} {cz-cc:>+8.3f}")
