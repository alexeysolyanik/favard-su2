"""Sharp constants h_k of Table 1 (and Table of state constants).

h_k is computed from the k node equations (moment-node-equations) by
Newton's method in 40-digit arithmetic, continuing the nodes from k-1 to k,
and the finite formula h_k = (2/pi) sum (-1)^(k-j) a_j cos^2 a_j.
The second block evaluates h_k^st = H(k,S^2)/4 from the explicit series.
"""
import mpmath as mp
mp.mp.dps = 40

def F(r, a):
    # primitive of sin^{2r} a cos^2 a
    return mp.quad(lambda s: mp.sin(s)**(2*r)*mp.cos(s)**2, [0, a])

def residuals(a, k):
    pts = [mp.mpf(0)] + list(a) + [mp.pi/2]
    res = []
    for r in range(k):
        s = 0
        for i in range(k+1):
            s += (-1)**(k-i)*(F(r, pts[i+1]) - F(r, pts[i]))
        res.append(s)
    return res

print("k   h_k")
prev = None
for k in range(1, 8):
    a0 = [mp.pi/8] if prev is None else [x*k/(k+0.5) for x in prev] + [(prev[-1]+mp.pi/2)/2]
    sol = mp.findroot(lambda *a: residuals(a, k), a0)
    a = [sol[i] for i in range(k)] if hasattr(sol, "rows") else [sol]
    h = 2/mp.pi*sum((-1)**(k-j)*a[j-1]*mp.cos(a[j-1])**2 for j in range(1, k+1))
    print(k, mp.nstr(h, 12))
    prev = a

print("\nk   h_k^st = H(k,S^2)/4")
for k in range(1, 6):
    H = 2*mp.nsum(lambda j: 1/((2*j+1)*((k+1)**2*(2*j+1)**2-1)), [0, mp.inf])
    print(k, mp.nstr(H/4, 8))
