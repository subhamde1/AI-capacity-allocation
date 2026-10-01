"""Separate implementation checks; not an independent human review or proof.
Run after long_budget.py. Integer trace accounting and independent MILP checks.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys,warnings,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
import extended_experiments as ex
P=Path(__file__).resolve().parent

def trace_audit():
    inp=json.loads((P/'trace_inputs.json').read_text())
    out=json.loads((P/'trace_results.json').read_text())
    assert out['input_sha256']==hashlib.sha256((P/'trace_inputs.json').read_bytes()).hexdigest()
    denom=inp['scale_E2'];cap=inp['capacity_C8'];calls=0;error=0.;checks=0
    # Recompute pseudo-random preferences without importing the study generator.
    def gen(seed,offset,width):
        z=seed;ans=[]
        for _ in range(1024):
            z=(1664525*z+1013904223)%4294967296
            ans.append(offset+width*z//4294967296)
        return np.array(ans,dtype=np.int64).reshape(128,8)
    vv=gen(20260929,6,19);cc=gen(20260922,0,5)
    for batch in inp['batches']:
        if batch['train']:continue
        A=np.zeros((8,4,36),dtype=np.int64);energy=[]
        for i,j in enumerate(batch['jobs']):
            energy.append(j['gpus']*j['duration'])
            for m in range(1,4):
                start=j['submit']+(m-1)*21600;end=start+j['duration']
                for t in range(36):
                    # Partition by integer seconds, algebraically independent form.
                    left=max(start,3600*t);right=min(end,3600*t+3600)
                    if right>left:A[i,m,t]=j['gpus']*(right-left)
                assert int(A[i,m].sum())==energy[-1]
        E=np.array(energy);rows=[r for r in out['records'] if r['date']==batch['date']]
        for r in rows:
            v=vv[r['r']];c=cc[r['r']]
            for k,(modes,wn) in enumerate(zip(r['modes'],r['w_num'])):
                m=np.array(modes);assert np.all((0<=m)&(m<=3))
                load=A[np.arange(8),m].sum(axis=0)
                assert np.all(8*load[:12 if k==6 else 36]<=cap)
                if k==1:assert np.all(m<3)
                if k==2:assert np.all(m<2)
                assert int(denom*(v*(m>0)).sum()-(c*E*np.maximum(m-1,0)).sum())==wn
                if k<2:
                    p=np.array(r['payment_full_num' if k==0 else 'payment_half_num'])
                    own=denom*v*(m>0)-c*E*np.maximum(m-1,0)
                    assert np.all(p>=0) and np.all(own-p>=0)
                    assert int(p.sum()+(own-p).sum())==wn
                checks+=1
        r=rows[0];v=vv[0];c=cc[0]
        coef=(denom*v[:,None]-c[:,None]*E[:,None]*np.arange(3))/denom
        matrix=np.vstack([np.repeat(np.eye(8),3,axis=1),8*A[:,1:,:].reshape(24,36).T])
        def solve(half=False,exclude=None):
            nonlocal calls
            upper=np.ones((8,3))
            if half:upper[:,2]=0
            if exclude is not None:upper[exclude,:]=0
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options detected')
                res=milp(-coef.ravel(),integrality=np.ones(24),bounds=Bounds(0,upper.ravel()),
                    constraints=LinearConstraint(matrix,-np.inf,np.r_[np.ones(8),np.full(36,cap)]),
                    options={'mip_rel_gap':0.,'threads':1,'random_seed':0})
            assert res.success
            calls+=1;return -float(res.fun)
        for h,k in [(False,0),(True,1)]:
            w=solve(h);error=max(error,abs(w-r['w_num'][k]/denom))
            modes=np.array(r['modes'][k]);own=denom*v*(modes>0)-c*E*np.maximum(modes-1,0)
            for i in range(8):
                pay=solve(h,i)-w+own[i]/denom
                saved=r['payment_half_num' if h else 'payment_full_num'][i]/denom
                error=max(error,abs(pay-saved))
    assert error<1e-8
    return {'allocation_checks':checks,'independent_milp_solves':calls,'max_absolute_difference':error}

def scale_audit():
    names=['scale_original_short','scale_paired_short','scale_paired_long']
    runs=[json.loads((P/(s+'.json')).read_text()) for s in names]
    checks=0;endpoints=0;maxviol=0.;maxobj=0.
    for dataset in runs:
        assert len(dataset)==45
        for r in dataset:
            a,q,C,v,c=ex.instance(*[r[k] for k in ['n','T','M','k','s']])
            assert np.max(np.ptp(a.sum(axis=2),axis=1))<1e-10
            for kind,s in [('full',r['full']),('half',r['half']),('firm',r['firm']),*enumerate(r['pivot'])]:
                y=np.array(s['y']);assert np.all((y==0)|(y==1)) and np.all(y.sum(axis=1)<=1)
                viol=float((np.einsum('im,imt->t',y,a)-C).max());maxviol=max(maxviol,viol)
                assert viol<1e-6
                if kind=='firm':assert y[:,1:].sum()==0
                if kind=='half':assert y[:,np.linspace(0,1,r['M'])>.5].sum()==0
                if isinstance(kind,int):assert y[kind].sum()==0
                val=float(((v[:,None]-c[:,None]*q)*y).sum())
                err=abs(val-s['L']);maxobj=max(maxobj,err);assert err<1e-7
                assert s['U']+1e-7>=s['L'] and abs(s['gap']-max(0,s['U']-s['L']))<1e-7
                checks+=1
            y=np.array(r['full']['y']);own=((v[:,None]-c[:,None]*q)*y).sum(axis=1)
            for i,p in enumerate(r['pivot']):
                for key,bound in [('payment_L','L'),('payment_U','U')]:
                    assert abs(r[key][i]-(p[bound]-r['full']['L']+own[i]))<1e-7
                    endpoints+=1
    for rs in zip(*runs):
        for kind in ['full','half','firm',*range(rs[0]['n'])]:
            sols=[r['pivot'][kind] if isinstance(kind,int) else r[kind] for r in rs]
            assert max(s['L'] for s in sols)<=min(s['U'] for s in sols)+1e-5
    return {'stored_solves_checked':checks,'payment_endpoints_checked':endpoints,
            'maximum_capacity_violation':maxviol,'maximum_objective_error':maxobj,
            'cross_run_bound_consistency':True,
            'limitation':'Checks primal records and bound consistency; does not independently prove solver dual bounds.'}

if __name__=='__main__':
    result={'scope':'Author-side computational audit using a separate implementation; not external certification.'}
    result['trace']=trace_audit();print(result,flush=True)
    if '--trace-only' not in sys.argv:result['scale']=scale_audit()
    (P/'independent_audit.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
