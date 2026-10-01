"""Frozen 45 instances; independent cold-start 2/.5 and 20/5 second budgets.
Six worker processes; one HiGHS thread per worker; no own-bid warm starts.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,sys,platform,warnings
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,scipy
import extended_experiments as e
from scipy.optimize import milp as original_milp
P=Path(__file__).resolve().parent
CASES=[(n,T,M,k,s) for n,T,M in [(8,4,2),(16,8,3),(32,12,5),(64,24,5),(128,24,8)]
       for k,s in product([0.,.5,1.],[.35,.6,.85])]

def single_thread_milp(*args,**kwargs):
    kwargs['options'].update(threads=1,random_seed=0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected')
        return original_milp(*args,**kwargs)
e.milp=single_thread_milp

def job(task):
    cid,kind=task;a,q,C,v,c=e.instance(*CASES[cid]);M=a.shape[1]
    kwargs={};limit=2.
    if kind=='half':kwargs['keep']=np.flatnonzero(np.linspace(0,1,M)<=.5)
    if kind=='firm':kwargs['keep']=[0]
    if isinstance(kind,int):kwargs['exclude']=kind;limit=.5
    return cid,kind,{label:e.solve(a,q,C,v,c,limit=limit*factor,**kwargs)
                    for label,factor in [('short',1),('long',10)]}

def assemble(records,label):
    rows=[]
    for cid,(n,T,M,k,s) in enumerate(CASES):
        full=records[cid]['full'][label];piv=[records[cid][str(i)][label] for i in range(n)]
        a,q,C,v,c=e.instance(n,T,M,k,s)
        own=((v[:,None]-c[:,None]*q)*np.array(full['y'])).sum(axis=1)
        rows.append({'n':n,'T':T,'M':M,'k':k,'s':s,'full':full,
          'half':records[cid]['half'][label],'firm':records[cid]['firm'][label],
          'pivot':piv,'pivot_sec':sum(x['sec'] for x in piv),
          'pivot_exact':sum(x['gap']<1e-6 for x in piv),'pivot_gapmax':max(x['gap'] for x in piv),
          'payment_L':(np.array([x['L'] for x in piv])-full['L']+own).tolist(),
          'payment_U':(np.array([x['U'] for x in piv])-full['L']+own).tolist()})
    return rows

if __name__=='__main__':
    target=P/'paired_tasks.jsonl';done={}
    if target.exists():
        for line in target.read_text().splitlines():
            cid,kind,r=json.loads(line);done[(cid,str(kind))]=r
    tasks=[(cid,kind) for cid,x in enumerate(CASES) for kind in ['full','half','firm',*range(x[0])]
           if (cid,str(kind)) not in done]
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool,target.open('a') as f:
        for fut in as_completed([pool.submit(job,t) for t in tasks]):
            cid,kind,r=fut.result();done[(cid,str(kind))]=r
            f.write(json.dumps([cid,kind,r],separators=(',',':'))+'\n');f.flush()
            if len(done)%50==0:print('completed',len(done),'of 2367',flush=True)
    records=[{} for _ in CASES]
    for (cid,kind),r in done.items():records[cid][kind]=r
    for label in ['short','long']:
        (P/('scale_paired_'+label+'.json')).write_text(json.dumps(assemble(records,label)))
    (P/'paired_environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,
      'scipy':scipy.__version__,'platform':platform.platform(),'workers':6,'highs_threads':1,
      'random_seed':0,'wall_seconds':time.perf_counter()-begin},indent=2))
    print('COMPLETE',len(done),flush=True)
