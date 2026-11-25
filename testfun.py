import numpy as np, math
exec(open('designs.py').read().split("for name, G in groups.items():")[0])
E = math.e
# quaternion q=(q0,q1,q2,q3); rotation matrix entries
def R(q):
    a,b,c,d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
f0 = lambda q: math.exp(R(q)[2,2]-1)/8
mean_f0 = (1-math.exp(-2))/16
c = (1,2,3); s = sum(c)
Fc = lambda q: sum(ci*math.exp(R(q)[i,i]) for i, ci in enumerate(c))/(8*E*s)
mean_Fc = mean_f0  # each diagonal entry has the law of t
# check dims (Fisher-type lower bound) and errors
for name, G in groups.items():
    e0 = abs(np.mean([f0(q) for q in G]) - mean_f0)
    ec = abs(np.mean([Fc(q) for q in G]) - mean_Fc)
    print(f"{name:18s} |f0 err|={e0:.6f}  |Fc err|={ec:.6f}")
print("dim V_{s+1}:", [(s+1)*(2*s+1)*(2*s+3)//3 for s in range(4)])
print("mean f0", mean_f0)
