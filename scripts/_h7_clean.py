import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, time
sys.path.insert(0,"scripts"); import riemann_gappair as r
gamma = np.array([float(x) for x in r.ZEROS.read_text().split()])
rng = np.random.default_rng(13)

def unfold(x, deg=6):
    nn=np.arange(len(x)); xs=(x-x[0])/(x[-1]-x[0]+1e-12)
    return np.polyval(np.polyfit(xs, nn, deg), xs)

def acorr(s):
    return float(np.corrcoef(np.diff(s)[:-1], np.diff(s)[1:])[0,1])

# CLEAN GUE: one large matrix, CONTIGUOUS bulk (middle 50%), high-quality local unfold. No concatenation.
def gue_bulk_clean(N):
    A=(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)))/np.sqrt(2)
    ev=np.sort(np.linalg.eigvalsh((A+A.conj().T)/np.sqrt(2)))
    return ev[N//4:3*N//4]                       # contiguous bulk, 1 sequence

print("CLEAN comparison — identical high-quality unfold on contiguous bulk sequences:",flush=True)
for deg in [4,6,8]:
    t0=time.time()
    # zeros: contiguous mid-range blocks of 4000, same unfold
    zc=[acorr(unfold(gamma[k:k+4000], deg)) for k in range(20000, 90000, 4000)]
    gc=[acorr(unfold(gue_bulk_clean(8000), deg)) for _ in range(12)]   # 12 matrices, bulk=4000 each
    zc,gc=np.array(zc),np.array(gc)
    print(f"  deg={deg}: zeros={zc.mean():+.3f}±{zc.std()/np.sqrt(len(zc)):.3f}  "
          f"GUE={gc.mean():+.3f}±{gc.std()/np.sqrt(len(gc)):.3f}  diff={zc.mean()-gc.mean():+.3f}"
          f"  ({time.time()-t0:.0f}s)",flush=True)
