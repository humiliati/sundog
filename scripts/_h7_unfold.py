import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, time
sys.path.insert(0,"scripts"); import riemann_gappair as r

gamma = np.array([float(x) for x in r.ZEROS.read_text().split()])   # RAW zeros
w_asym = r.load_unfolded()                                          # asymptotic ⟨N⟩ unfold
rng = np.random.default_rng(11)

def unfold_local(x, deg):
    """per-block: unfold a sorted sub-sequence by a degree-`deg` poly fit of rank vs x (identical for all)."""
    nn = np.arange(len(x)); xs = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    return np.polyval(np.polyfit(xs, nn, deg), xs)

def corr(seqs):
    cs = [np.corrcoef(np.diff(s)[:-1], np.diff(s)[1:])[0,1] for s in seqs if len(s) > 3]
    return float(np.mean(cs)), float(np.std(cs)/np.sqrt(max(len(cs),1)))

def gue_eig(N):
    A = (rng.standard_normal((N,N)) + 1j*rng.standard_normal((N,N)))/np.sqrt(2)
    ev = np.sort(np.linalg.eigvalsh((A+A.conj().T)/np.sqrt(2)))
    return ev[int(N*0.25):int(N*0.75)]      # bulk (raw eigenvalues)

L = 1000
print("IDENTICAL local unfolding (deg-d poly of rank-vs-x per block) applied to BOTH zeros and GUE:",flush=True)
print(f"  {'deg':>4} {'zeros':>15} {'GUE':>15} {'diff':>8}",flush=True)
# GUE raw bulk eigenvalues, pooled then blocked (generate enough for ~40 blocks)
gue_raw = np.concatenate([gue_eig(2200) for _ in range(40)])  # ~44000 bulk levels
for deg in [2, 3, 5, 8]:
    t0=time.time()
    zb = [unfold_local(gamma[i*L:(i+1)*L], deg) for i in range(len(gamma)//L)]
    cz, ez = corr(zb)
    # block the pooled GUE bulk into L-chunks (each chunk is locally smooth)
    gb = [unfold_local(gue_raw[i*L:(i+1)*L], deg) for i in range(len(gue_raw)//L)]
    cg, eg = corr(gb)
    print(f"  {deg:>4} {cz:>7.3f}±{ez:.3f} {cg:>7.3f}±{eg:.3f} {cz-cg:>+8.3f}   ({time.time()-t0:.0f}s)",flush=True)
print(f"\n  (reference: asymptotic ⟨N⟩ unfold of zeros gave -0.357; GUE semicircle-bulk gave -0.31)",flush=True)
