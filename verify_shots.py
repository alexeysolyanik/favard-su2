import numpy as np, math
src = open('compression_example.py').read().split('# 1. Lemma check')[0]
exec(src)
exec(open('designs.py').read().split("for name, G in groups.items():")[0])
G = np.array(groups['icosahedral (2I)'])
z, w = to_zw(G)
f, ft = response(W, z, w), response(Wc, z, w)
d = ft - f
EY = np.mean(d**2)
EY2 = np.mean((2 - 2*f*ft)**2)          # E[Y^2 | U] for Y=(X~-X)(X~'-X'), averaged over commands
var = EY2 - np.mean(d**2)**2           # variance of one Y under uniform command choice (upper bound use)
print("mean d^2", EY, "E Y^2", EY2, "sd Y", math.sqrt(var))
for zz in (2, 3):
    print(f"pairs for {zz} s.e. = mean d^2: {(zz*math.sqrt(var)/EY)**2:.3e}")
beta = 0.05
print("Hoeffding pairs for radius = mean d^2:", 64*math.log(2/beta)/(2*EY**2))
print("single-command shots for 0.0125 at 3 sigma:", 2*9/0.0125**2)
