
"""Self-contained extensions. No downloaded data; all input formulas are below.
Run with Python 3.12, NumPy 2.3.5, SciPy 1.17.0. Outputs JSON.
"""
import json,time,platform,sys
from pathlib import Path
from itertools import product
import numpy as np
import scipy
from scipy.optimize import milp,Bounds,LinearConstraint,linear_sum_assignment

ROOT=Path(__file__).resolve().parent

def lcg(seed,count):
    out=[]
    for _ in range(count):
        seed=(1664525*seed+1013904223)%(2**32)
        out.append(seed)
    return np.array(out,dtype=np.uint64)

def instance(n,T,M,kappa,scarcity):
    # n,T,M are coordinated size configurations, not separate factorial effects.
    S=T//2; R=T-S
    z=lcg(20260928+n+100*T+10000*M,5*n).reshape(n,5)
    d=2+((5*z[:,0])//(2**32)).astype(int)
    v=10+((21*z[:,1])//(2**32)).astype(int)
    theta=((5*z[:,2])//(2**32)).astype(int)
    # Two stress pulses on neighboring cyclic indices; energy split 3/4 and 1/4.
    ref=np.zeros((n,T));ker=np.zeros((n,R))
    for i in range(n):
        s=i%S;ref[i,s]+=3*d[i]/4;ref[i,(s+1)%S]+=d[i]/4
        r=int((R*z[i,3])//(2**32))
        width=1+int((min(3,R)*z[i,4])//(2**32))
        for j in range(width):ker[i,(r+j)%R]=1/width
    ker=(1-kappa)*ker
    ker[:,0]+=kappa
    f=np.linspace(0,1,M)
    a=np.repeat(ref[:,None,:],M,axis=1)*(1-f[None,:,None])
    a[:,:,S:]+=d[:,None,None]*f[None,:,None]*ker[:,None,:]
    q=d[:,None]*f[None,:]
    # Same total capacity in each half; nonuniform public temporal bottlenecks.
    g=1+.2*((np.arange(T)%3)-1)
    C=np.r_[scarcity*d.sum()*g[:S]/g[:S].sum(),
            scarcity*d.sum()*g[S:]/g[S:].sum()]
    assert np.max(abs(a.sum(axis=2)-d[:,None]))<1e-10
    return a,q,C,v,theta

def solve(a,q,C,v,c,keep=None,exclude=None,limit=2.):
    n,M,T=a.shape
    idx=np.arange(n*M)
    if keep is not None:idx=idx[np.tile(np.isin(np.arange(M),keep),n)]
    if exclude is not None:idx=idx[idx//M!=exclude]
    A=np.vstack([np.eye(n)[idx//M].T,a.reshape(-1,T)[idx].T])
    coef=(v[:,None]-c[:,None]*q).ravel()[idx]
    start=time.perf_counter()
    res=milp(-coef,integrality=np.ones(len(idx)),bounds=Bounds(0,1),
      constraints=LinearConstraint(A,-np.inf,np.r_[np.ones(n),C]),
      options={'mip_rel_gap':0.,'time_limit':limit,'presolve':True})
    elapsed=time.perf_counter()-start
    # All-rejection is a known feasible incumbent if no solver incumbent exists.
    y=np.zeros(n*M,dtype=int)
    if res.x is not None:y[idx]=(res.x>.5).astype(int)
    assert np.all(y.reshape(n,M).sum(axis=1)<=1)
    assert np.max(a.reshape(-1,T).T@y-C)<=1e-6
    val=float((v[:,None]-c[:,None]*q).ravel()@y)
    dual=getattr(res,'mip_dual_bound',None)
    # Trivial feasible-relaxation upper bound also covers a missing dual bound.
    available=np.arange(n)!=exclude if exclude is not None else np.ones(n,dtype=bool)
    relaxed=np.maximum(0,(v[:,None]-c[:,None]*q).max(axis=1))
    upper=float(relaxed[available].sum())
    if dual is not None and np.isfinite(dual):upper=min(upper,-float(dual))
    upper=max(upper,val)
    return {'L':val,'U':upper,'gap':max(0.,upper-val),'sec':elapsed,
      'admit':int(y.sum()),'flex':int(y.reshape(n,M)[:,1:].sum()),
      'status':int(res.status),'y':y.reshape(n,M).tolist()}

def scalability():
    rows=[]
    configs=[(8,4,2),(16,8,3),(32,12,5),(64,24,5),(128,24,8)]
    for n,T,M in configs:
      for k,s in product([0.,.5,1.],[.35,.6,.85]):
        a,q,C,v,c=instance(n,T,M,k,s)
        full=solve(a,q,C,v,c)
        half=solve(a,q,C,v,c,keep=np.flatnonzero(np.linspace(0,1,M)<=.5))
        firm=solve(a,q,C,v,c,keep=[0])
        piv=[solve(a,q,C,v,c,exclude=i,limit=.5) for i in range(n)]
        own=(v[:,None]-c[:,None]*q)*np.array(full['y'])
        own=own.sum(axis=1)
        pl=np.array([x['L'] for x in piv])-(full['L']-own)
        pu=np.array([x['U'] for x in piv])-(full['L']-own)
        row={'n':n,'T':T,'M':M,'k':k,'s':s,'full':full,'half':half,
          'firm':firm,'pivot_sec':sum(x['sec'] for x in piv),
          'pivot_gapmax':max(x['gap'] for x in piv),
          'pivot_exact':sum(x['gap']<1e-6 for x in piv),
          'pivot':piv,'payment_L':pl.tolist(),'payment_U':pu.tolist()}
        rows.append(row)
        (ROOT/'scale_results.json').write_text(json.dumps(rows,indent=2))
        print(n,k,s,round(full['L'],3),round(full['gap'],5),
          round(row['pivot_sec'],2),row['pivot_exact'],flush=True)
    return rows

def economic():
    # New independent value vectors; original costs and physical model unchanged.
    D=np.array([2,4,2,4,4,2,4,2]);H=np.array([0,1,1,0,0,1,1,0])
    modes=np.array(list(product(range(4),repeat=8)),dtype=np.int8)
    admitted=modes>0
    Q=np.choose(modes,[np.zeros(8),np.zeros(8),D/2,D])
    masks=admitted@(1<<np.arange(8))
    costs=((5*lcg(20260922,1024))//(2**32)).astype(int).reshape(128,8)
    vals=(6+(19*lcg(20260929,1024))//(2**32)).astype(int).reshape(128,8)
    out=[]
    for k,kp,kr,scale in product([0.,.5,1.],[6],[3,6,9],[.5,1.,2.]):
      alpha=k+(1-k)*H
      load=np.zeros((len(modes),4))
      work=D*admitted-Q
      load[:,0]=work[:,::2].sum(axis=1);load[:,1]=work[:,1::2].sum(axis=1)
      load[:,2]=Q@alpha;load[:,3]=Q@(1-alpha)
      good=np.flatnonzero(np.all(load<=[kp,kp,kr,kr],axis=1))
      half=good[np.all(modes[good]<3,axis=1)]
      for r in range(128):
        v=vals[r];c=scale*costs[r];w=admitted@v-Q@c
        j=good[np.argmax(w[good])];jh=half[np.argmax(w[half])]
        g=np.full(256,-np.inf);np.maximum.at(g,masks[good],w[good])
        seq=[]
        for order in [sorted(range(8),key=lambda i:(c[i],i)),sorted(range(8),key=lambda i:(-v[i],i))]:
          mask=0
          for i in order:
            trial=mask|(1<<i)
            if g[trial]>=g[mask]:mask=trial
          seq.append(g[mask])
        pairs=sum(v[i]>v[jj] and not admitted[j,i] and admitted[j,jj]
                  for i in range(8) for jj in range(8))
        out.append({'k':k,'kr':kr,'scale':scale,'r':r,'W':float(w[j]),
          'half_loss':float(100*(w[j]-w[jh])/w[j]),
          'cost_loss':float(100*(w[j]-seq[0])/w[j]),
          'value_loss':float(100*(w[j]-seq[1])/w[j]),
          'reversal':int(pairs>0),'admit':int(admitted[j].sum())})
    (ROOT/'economic_results.json').write_text(json.dumps(out,indent=2))
    # Isolate an incompatible broad profile in Theorem 3 with r=3.
    # Narrow v=10 and theta=1 each, broad v=x, theta=c; K=1/3 or 2/3.
    # With tight recovery: broad-alone versus 3 narrows; with loose, all fit.
    grid=[]
    for cap,x,c in product([1/3,2/3],[10,20,28,30,40],[0,2,6]):
      profiles=np.array([[1/3,1/3,1/3,0],
        [1/3,0,0,2/3],[0,1/3,0,2/3],[0,0,1/3,2/3]])
      vals4=np.array([x,10,10,10]);cost4=np.array([c,1,1,1])
      vv=np.array(list(product([0,1],repeat=4)))
      good=np.all(vv@profiles<=[cap,cap,cap,2],axis=1)
      w=vv@(vals4-cost4);j=np.argmax(np.where(good,w,-np.inf))
      grid.append([cap,x,c,int(vv[j,0]),float(w[j]),vv[j].tolist()])
    (ROOT/'broad_value_grid.json').write_text(json.dumps(grid,indent=2))
    print('economic cases',len(out),flush=True)

def assignment(W,C):
    # W[i,t] is net value or -infinity if ineligible; n dummy rejection slots.
    slots=np.repeat(np.arange(len(C)),np.minimum(C,len(W)))
    a=np.c_[W[:,slots],np.zeros((len(W),len(W)))]
    a=np.where(np.isfinite(a),a,-1e9)
    i,j=linear_sum_assignment(a,maximize=True)
    alloc=np.full(len(W),-1,dtype=int)
    use=j<len(slots);alloc[i[use]]=slots[j[use]]
    return float(a[i,j].sum()),alloc

def chain_prices(W,C,alloc):
    # Nodes: periods and sink. Edges reverse displaced assignments.
    T=len(C);dist=np.full((T+1,T+1),np.inf);np.fill_diagonal(dist,0)
    for t in range(T):
      if np.sum(alloc==t)<C[t]:dist[t,T]=0
      for j in np.flatnonzero(alloc==t):
        dist[t,T]=min(dist[t,T],W[j,t])
        for u in range(T):
          if np.isfinite(W[j,u]):dist[t,u]=min(dist[t,u],W[j,t]-W[j,u])
    for k in range(T+1):dist=np.minimum(dist,dist[:,k,None]+dist[None,k,:])
    assert np.min(np.diag(dist))>=-1e-8
    return dist[:T,T]

def structural_audit():
    worst=0.;checks=0;greedy_gap=0.;enum_gap=0.;rows=[]
    for seed in range(100):
      n,T=6,4
      z=lcg(111+seed, n*T+2*n+T).astype(int)
      eligible=(z[:n*T].reshape(n,T)%3)>0
      v=5+z[n*T:n*T+n]%16;c=z[n*T+n:n*T+2*n]%5
      q=np.tile(np.arange(T),(n,1))
      C=1+z[-T:]%2
      W=np.where(eligible,v[:,None]-c[:,None]*q,-np.inf)
      w,alloc=assignment(W,C);prices=chain_prices(W,C,alloc)
      choices=np.array(list(product(range(-1,T),repeat=n)))
      feasible=np.all(np.array([(choices==t).sum(axis=1) for t in range(T)]).T<=C,axis=1)
      extended=np.c_[np.zeros(n),W]
      welfare=extended[np.arange(n)[None,:],choices+1].sum(axis=1)
      enumerated=float(np.max(np.where(feasible,welfare,-np.inf)))
      enum_gap=max(enum_gap,abs(w-enumerated))
      for t in range(T):
        cm=C.copy();cm[t]-=1
        wminus,_=assignment(W,cm)
        worst=max(worst,abs(prices[t]-(w-wminus)));checks+=1
      # Slot-invariant scores: greedy with full rematching.
      score=v-c;constant=np.where(eligible,score[:,None],-np.inf)
      optimum,_=assignment(constant,C);selected=[]
      for i in sorted(range(n),key=lambda i:(-score[i],i)):
        if score[i]<0:continue
        trial=selected+[i]
        weight=np.where(eligible[trial],1.,-np.inf)
        wtest,_=assignment(weight,C)
        if wtest==len(trial):selected=trial
      greedy_gap=max(greedy_gap,abs(optimum-score[selected].sum()))
    # Chained displacement example: u1 values 10 at A,8 at B;
    # u2 values 9 at B,8 at C; u3 value 4 at C. Capacity one.
    W=np.array([[10,8,-np.inf],[-np.inf,9,8],[-np.inf,-np.inf,4.]])
    C=np.ones(3,dtype=int);w,a=assignment(W,C)
    p=chain_prices(W,C,a)
    out={'chain_checks':checks,'max_chain_error':worst,'greedy_checks':100,
      'max_greedy_error':greedy_gap,'enumerated_allocations':100*5**6,
      'max_matching_enumeration_error':enum_gap,'chain_example':p.tolist(),'W':w}
    # A scarce slot can be released by several reallocations, not just eviction.
    (ROOT/'structural_audit.json').write_text(json.dumps(out,indent=2));print(out,flush=True)

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    (ROOT/'environment.json').write_text(json.dumps({'python':sys.version,
      'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},indent=2))
    if mode in ['structural','all']:structural_audit()
    if mode in ['economic','all']:economic()
    if mode in ['scale','all']:scalability()
