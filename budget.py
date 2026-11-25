"""Query budgets of Appendix B (Table of budgets and crossing intervals).

For each tolerance tau: mesh degree L, fine cubature size J, quantum query
budget Q with the simple choice b_L=3/(4k^2) and with the certified h_k,
and the classical Monte Carlo budget N_MC. Then the intervals of tau where
the quantum budget is smaller, and tau_q for the budgets in the text.
"""
from fractions import Fraction as F
import math
hbar={1:F('0.22153761035678'),2:F('0.08646617012829'),3:F('0.04673319581883'),4:F('0.02938540963012'),5:F('0.02021569929310')}
PI=math.pi
def L_of(tau): return 2*math.ceil(float(tau)**(-0.2)/2)
def run(coef,tau,pi=PI):
    tau=F(tau); L=L_of(tau); k=L//2+1
    b=lambda n: coef/F((n//2+1)**2)
    K=L
    while not 2*b(K)<=tau/4: K+=2
    D=K+2;n=D+1;r=D//2+1;J=r*n*n
    d=tau/8
    out={}
    for name,bL in (('simple',b(L)),('cert',hbar[k])):
        R0=5*(bL+d); M=2
        while not 2*float(R0)*(pi/M+pi**2/M**2)<=float(tau)/4: M*=2
        out[name]=(4*(10*L+1)**3+4*M-2,M,R0)
    p0=8/PI**2; h1=float(hbar[1])
    NMC=math.ceil((2*h1+float(d))**2/(3*(1-p0)*(float(tau)-float(d)/2)**2))
    return L,K,J,out,NMC
taus=['1/256','1/1000','9/10000','6/10000','1/10000']
for coef,lab in ((F(3,4),'b_L = 3/(4k^2)'),):
    print(lab)
    for t in taus:
        L,K,J,o,N=run(coef,t)
        print(t,L,K,J,'simple',o['simple'][0],o['simple'][1],'cert',o['cert'][0],o['cert'][1],N)
    print('R0 simple 1/256:',run(coef,'1/256')[3]['simple'][2])

import numpy as np
def Q(coef,tau,which):
    L=L_of(tau); k=L//2+1
    bL=coef/F(k*k) if which=='simple' else hbar[k]
    d=tau/8; R0=5*(float(bL)+d); M=2
    while not 2*R0*(PI/M+PI**2/M**2)<=tau/4: M*=2
    return 4*(10*L+1)**3+4*M-2
def NMC(tau):
    p0=8/PI**2; h1=float(hbar[1]); d=tau/8
    return math.ceil((2*h1+d)**2/(3*(1-p0)*(tau-d/2)**2))
def tauq(q):
    p0=8/PI**2; return 2*float(hbar[1])/((15/16)*math.sqrt(3*(1-p0)*q)-1/8)
for coef,lab in ((F(3,4),'b_L = 3/(4k^2)'),):
  for which in ('cert','simple'):
    ts=np.geomspace(1e-4,1,400000)
    good=[Q(coef,t,which)<NMC(t) for t in ts]
    iv=[];start=None
    for t,g in zip(ts,good):
        if g and start is None: start=t
        if not g and start is not None: iv.append((start,prev)); start=None
        prev=t
    if start is not None: iv.append((start,ts[-1]))
    print(lab,which,[(f'{a:.9g}',f'{b:.9g}',Q(coef,(a+b)/2,which)) for a,b in iv])
for q in (308450,940690,406754,1038994,341218,973458,940690,292066,2387906): print(q,tauq(q))
