"""Certifying a compressed circuit: a 4-qubit Ising example.

Original:  U on qubit 0, then W = 40 first-order Trotter steps of
           H = J sum Z_i Z_{i+1} + h sum X_i  (time T), then U on qubit 0,
           measure Z on qubit 0.
Compressed: W replaced by a 3-layer Trotter-shaped ansatz with free angles,
           fitted to W.  The command U is kept as a gate, so both responses
           have spin <= 2 (Lemma spin-count) and d = f~ - f lies in V_3.
"""
import numpy as np, itertools, math, time
from scipy.optimize import minimize
from scipy.linalg import expm

n = 4; dim = 2**n
J, hx, T = 1.0, 1.0, 1.0
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1., -1.])

def kron(*ops):
    out = np.array([[1.0]])
    for o in ops: out = np.kron(out, o)
    return out

def op_on(o, q):
    return kron(*[o if i == q else I2 for i in range(n)])

ZZ = [op_on(Z, i) @ op_on(Z, i+1) for i in range(n-1)]
Xs = [op_on(X, i) for i in range(n)]
zz_diag = [np.diag(m).real for m in ZZ]

def layer(zz_angles, x_angles):
    # exp(-i sum a_b ZZ_b) then exp(-i sum c_q X_q)
    D = np.exp(-1j*sum(a*z for a, z in zip(zz_angles, zz_diag)))
    Ux = kron(*[expm(-1j*c*X) for c in x_angles])
    return Ux @ np.diag(D)

def trotter(steps, Ttot=T):
    dt = Ttot/steps
    L = layer([J*dt]*(n-1), [hx*dt]*n)
    return np.linalg.matrix_power(L, steps)

W = trotter(40)
H = J*sum(ZZ) + hx*sum(Xs)
print("Trotter-40 vs exact exp(-iHT): ||W - e^{-iHT}|| =", np.linalg.norm(W - expm(-1j*H*T), 2))

NL = 3
def ansatz(p):
    p = p.reshape(NL, (n-1)+n)
    U = np.eye(dim, dtype=complex)
    for row in p:
        U = layer(row[:n-1], row[n-1:]) @ U
    return U

def infid(p):
    return 1 - abs(np.trace(ansatz(p).conj().T @ W))/dim

p0 = np.concatenate([[J*T/NL]*(n-1) + [hx*T/NL]*n for _ in range(NL)])
print("plain 3-step Trotter infidelity:", infid(p0))
res = minimize(infid, p0, method='BFGS', options={'gtol': 1e-12, 'maxiter': 5000})
Wc = ansatz(res.x)
print("optimized 3-layer infidelity:", res.fun, "operator distance", min(np.linalg.norm(Wc*np.exp(1j*a) - W, 2) for a in np.linspace(0, 2*np.pi, 721)))
cnot = lambda layers: layers*2*(n-1)
print("CNOT counts: original", cnot(40), "compressed", cnot(NL))

O = op_on(Z, 0)
def response(Wmat, zq, wq):
    # batch over commands U(z,w); qubit 0 is the most significant
    U = np.stack([np.stack([zq, -np.conj(wq)], -1), np.stack([wq, np.conj(zq)], -1)], -2)  # (B,2,2)
    psi1 = np.zeros((len(zq), dim), complex); psi1[:, 0] = zq; psi1[:, dim//2] = wq
    psi2 = psi1 @ Wmat.T
    psi3 = np.einsum('bij,bjk->bik', U, psi2.reshape(-1, 2, dim//2)).reshape(-1, dim)
    return np.einsum('bi,ij,bj->b', psi3.conj(), O, psi3).real

def to_zw(x):
    return x[:, 0] + 1j*x[:, 1], x[:, 2] + 1j*x[:, 3]

def d_of(x):
    z, w = to_zw(x)
    return response(Wc, z, w) - response(W, z, w)

rng = np.random.default_rng(271828)
def haar(N):
    y = rng.standard_normal((N, 4)); return y/np.linalg.norm(y, axis=1, keepdims=True)

# 1. Lemma check: f and d lie in V_3 = span of degree-4 monomials on S^3
mons = [a for a in itertools.product(range(5), repeat=4) if sum(a) == 4]
def basis(x): return np.stack([np.prod(x**np.array(a), axis=1) for a in mons], 1)
xs = haar(4000); B = basis(xs)
for name, vals in (("f", response(W, *to_zw(xs))), ("d", d_of(xs))):
    c, *_ = np.linalg.lstsq(B, vals, rcond=None)
    print(f"degree-4 fit residual for {name}: {np.abs(B@c - vals).max():.2e}  (dim {len(mons)})")
cd, *_ = np.linalg.lstsq(B, d_of(xs), rcond=None)

# 2. Uniform certificate on the grid of Appendix B with L = 4, q = 40
L = 4; q = 10*L
vals = -1 + 2*np.arange(q+1)/q
t0 = time.time(); best = 0.0; npts = 0
for chart in range(4):
    g = np.stack(np.meshgrid(vals, vals, vals, indexing='ij'), -1).reshape(-1, 3)
    v = np.insert(g, chart, 1.0, axis=1)
    x = v/np.linalg.norm(v, axis=1, keepdims=True)
    best = max(best, np.abs(d_of(x)).max()); npts += len(x)
print(f"grid nodes {npts}, max|d| on grid {best:.6e}, certificate 2*max = {2*best:.6e}, time {time.time()-t0:.1f}s")

# true sup of |d|: local maximization of the fitted polynomial from many starts
def negabs(y):
    x = y/np.linalg.norm(y); return -abs(basis(x[None])[0] @ cd)
starts = haar(400); sup = 0
for s in starts:
    r = minimize(negabs, s, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 4000})
    sup = max(sup, -r.fun)
xd = haar(2_000_000); dmc = d_of(xd)
print(f"sup|d| (optimized) {sup:.6e}; max over 2e6 Haar points {np.abs(dmc).max():.6e}")

# 3. Exact mean and mean square by designs
exec(open('designs.py').read().split("for name, G in groups.items():")[0])
for name, G in groups.items():
    dv = d_of(np.array(G))
    print(f"{name:18s} |G|={len(G):3d}: mean d = {dv.mean():+.6e}, mean d^2 = {np.mean(dv**2):.6e}")
print(f"Haar Monte Carlo (2e6): mean d = {dmc.mean():+.6e} +- {dmc.std()/np.sqrt(len(dmc)):.1e}, mean d^2 = {np.mean(dmc**2):.6e} +- {np.std(dmc**2)/np.sqrt(len(dmc)):.1e}")
# exact Haar moments from the fitted polynomial via the monomial moment formula
def mom(a):
    if any(ai % 2 for ai in a): return 0.0
    b = [ai//2 for ai in a]
    num = np.prod([math.gamma(bi+0.5)/math.gamma(0.5) for bi in b]); den = math.gamma(2+sum(b))/math.gamma(2)
    return num/den
M1 = np.array([mom(a) for a in mons])
G2 = np.array([[mom(tuple(np.add(a, b))) for b in mons] for a in mons])
print(f"exact Haar from polynomial: mean d = {cd@M1:+.6e}, mean d^2 = {cd@G2@cd:.6e}")
fW = response(W, *to_zw(xd))
print(f"size of the original response: max|f| {np.abs(fW).max():.4f}, RMS {np.sqrt(np.mean(fW**2)):.4f}")
