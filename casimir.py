import numpy as np
from scipy.linalg import expm
rng = np.random.default_rng(1)
def basis(d):
    B = []
    for a in range(d):
        for b in range(a+1, d):
            M = np.zeros((d,d), complex); M[a,b]=M[b,a]=1; B.append(M)
            M = np.zeros((d,d), complex); M[a,b]=-1j; M[b,a]=1j; B.append(M)
    for k in range(1, d):
        M = np.zeros((d,d), complex)
        M[:k,:k] = np.eye(k); M[k,k] = -k
        B.append(M*np.sqrt(2/(k*(k+1))))
    return [1j*L for L in B]   # Tr(L_a L_b)=2 delta, so <X,Y>=Re Tr(X*Y)/2 is orthonormal
def herm_traceless(d):
    A = rng.standard_normal((d,d))+1j*rng.standard_normal((d,d)); A = A+A.conj().T
    return A - np.trace(A)/d*np.eye(d)
for d in (2, 3, 4, 8):
    O, rho = herm_traceless(d), herm_traceless(d)
    A = rng.standard_normal((d,d))+1j*rng.standard_normal((d,d))
    U = expm(1j*(A+A.conj().T)/2)
    f = lambda V: np.trace(O@V@rho@V.conj().T).real
    h = 1e-4; lap = 0
    for X in basis(d):
        lap += (f(expm(h*X)@U) - 2*f(U) + f(expm(-h*X)@U))/h**2
    print(d, round(-lap/f(U), 4), 4*d)
