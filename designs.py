import numpy as np, itertools, math
from math import factorial

phi = (1+5**0.5)/2

def unique(qs):
    out = []
    for q in qs:
        q = np.array(q, float)
        if not any(np.allclose(q, p) for p in out):
            out.append(q)
    return np.array(out)

def q8():
    qs = []
    for i in range(4):
        for s in (1, -1):
            v = [0]*4; v[i] = s; qs.append(v)
    return qs

def binary_T():
    qs = q8()
    for s in itertools.product((1, -1), repeat=4):
        qs.append([x/2 for x in s])
    return unique(qs)

def binary_O():
    qs = list(binary_T())
    for i, j in itertools.combinations(range(4), 2):
        for s1, s2 in itertools.product((1, -1), repeat=2):
            v = [0]*4; v[i] = s1/2**0.5; v[j] = s2/2**0.5; qs.append(v)
    return unique(qs)

def even_perms(n=4):
    out = []
    for p in itertools.permutations(range(n)):
        inv = sum(1 for a in range(n) for b in range(a+1, n) if p[a] > p[b])
        if inv % 2 == 0: out.append(p)
    return out

def binary_I():
    qs = list(binary_T())
    base = [phi/2, 0.5, 1/(2*phi), 0.0]
    for p in even_perms():
        for s in itertools.product((1, -1), repeat=3):
            v = [0.0]*4
            vals = [base[0]*s[0], base[1]*s[1], base[2]*s[2], 0.0]
            for k in range(4): v[p[k]] = vals[k]
            qs.append(v)
    return unique(qs)

groups = {'Pauli (Q8)': unique(q8()), 'tetrahedral (2T)': binary_T(),
          'Clifford (2O)': binary_O(), 'icosahedral (2I)': binary_I()}

def qmul(a, b):
    a0,a1,a2,a3 = a; b0,b1,b2,b3 = b
    return np.array([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2,
                     a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0])

for name, G in groups.items():
    # closure check
    ok = all(any(np.allclose(qmul(a, b), c) for c in G) for a in G for b in G)
    # invariant dimension of spin j (SO(3) rep) = average of character over group
    # rotation angle of quaternion q: cos(angle/2) = q0
    dims = []
    for j in range(1, 13):
        s = 0
        for q in G:
            a = 2*math.acos(max(-1, min(1, q[0])))
            s += (2*j+1) if abs(math.sin(a/2)) < 1e-12 else math.sin((2*j+1)*a/2)/math.sin(a/2)
        dims.append(round(s/len(G), 6))
    t = next(j for j, d in enumerate(dims, 1) if abs(d) > 1e-9) - 1
    print(f"{name}: |G|={len(G)} (projective {len(G)//2}), closed={ok}, t={t}, inv dims j=1..12: {dims}")
