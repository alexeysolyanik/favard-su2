import numpy as np, sys
sys.path.insert(0, '.')
exec(open('designs.py').read().split("for name, G in groups.items():")[0])

def K(u):
    u = np.clip(u, 0, 1-1e-16)
    s = np.sqrt(u)
    return s*np.arcsin(s)/(2*np.sqrt(1-u)) - 0.25

rng = np.random.default_rng(20261124)
def haar(n):
    y = rng.standard_normal((n, 4)); return y/np.linalg.norm(y, axis=1, keepdims=True)

N = 100_000_000
chunk = 1_000_000
sets = {'single point (t=0)': np.array([[1.0, 0, 0, 0]])}
for k, v in groups.items():
    # projective representatives: keep q with first nonzero coordinate positive
    reps = [q for q in v if q[np.nonzero(np.abs(q) > 1e-12)[0][0]] > 0]
    sets[k] = np.array(reps)
for name, G in sets.items():
    vals = np.empty(N)
    for s in range(0, N, chunk):
        Y = haar(chunk)
        vals[s:s+chunk] = K((Y@G.T)**2).mean(axis=1)
    c = np.median(vals)
    a = np.abs(vals - c)
    print(f"{name:22s} m={len(G):3d}  e={a.mean():.5f} +- {a.std()/np.sqrt(N):.5f}  median={c:.5f}  mean(Phi)={vals.mean():.5f}")
