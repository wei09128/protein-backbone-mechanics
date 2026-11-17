import numpy as np, itertools
from scipy import stats
rng=np.random.default_rng(7)
def cov(G,k,s,r,tau,B=2000,R=600,sdL=0.5):
    c_p=c_t=0
    for _ in range(R):
        v=rng.normal(0,sdL,(G,k)); v-=v.mean(1,keepdims=True)
        si=s+rng.normal(0,tau,G); se=s*sdL*np.sqrt(1/r**2-1)
        q=si[:,None]*v+rng.normal(0,se,(G,k)); q-=q.mean(1,keepdims=True)
        sxy,sxx=(q*v).sum(1),(v*v).sum(1); sh=sxy.sum()/sxx.sum()
        W=rng.multinomial(G,np.ones(G)/G,size=B); bs=(W@sxy)/(W@sxx)
        lo,hi=np.quantile(bs,[.025,.975]); c_p+=lo<=s<=hi
        h=stats.t.ppf(.975,G-1)*bs.std(ddof=1); c_t+=sh-h<=s<=sh+h
    return c_p/R,c_t/R
print('G k s r tau | percentile  t(G-1)-SE')
for G,k,s,r,tau in itertools.product([20,12],[4],[1.0],[0.5,0.75],[0.0,0.2]):
    p,t=cov(G,k,s,r,tau); print(G,k,s,r,tau,'|',round(p,3),round(t,3),flush=True)
