"""Philly workload profiles. Observed GPU reservations, assumed preferences/modes.
Extract: python trace_robustness.py extract cluster_job_log.json
Analyze: python trace_robustness.py analyze
"""
import json,sys,hashlib
from pathlib import Path
from datetime import datetime,timedelta
from itertools import product
import numpy as np
from extended_experiments import lcg
P=Path(__file__).resolve().parent
def hourly(s,d,g,h=0):
    return np.array([g*max(0,min(3600*(t+1),s+h+d)-max(3600*t,s+h))
                     for t in range(36)],dtype=np.int64)

def extract(source):
    rs=json.loads(Path(source).read_text());first=datetime(2017,9,25)
    buckets={str((first+timedelta(days=j)).date()):[] for j in range(28)}
    counts={'total_jobs':len(rs),'in_dates':0,'status_pass':0,'single_attempt':0,
            'valid_times_resources':0,'eligible':0}
    for r in rs:
        try:s=datetime.fromisoformat(r['submitted_time'])
        except (KeyError,TypeError,ValueError):continue
        key=str(s.date())
        if key not in buckets:continue
        counts['in_dates']+=1
        if r.get('status')!='Pass':continue
        counts['status_pass']+=1
        if len(r.get('attempts',[]))!=1:continue
        counts['single_attempt']+=1;a=r['attempts'][0]
        try:
            st=datetime.fromisoformat(a['start_time']);end=datetime.fromisoformat(a['end_time'])
            g=len({(x['ip'],j) for x in a['detail'] for j in x['gpus']})
        except (KeyError,TypeError,ValueError):continue
        duration=int((end-st).total_seconds())
        if st<s or not 0<duration<=21600 or not 1<=g<=8:continue
        counts['valid_times_resources']+=1
        midnight=s.replace(hour=0,minute=0,second=0,microsecond=0)
        submit=int((s-midnight).total_seconds())
        if not 0<=submit<43200:continue
        counts['eligible']+=1
        buckets[key].append({'id':r['jobid'],'submit':submit,'start':int((st-midnight).total_seconds()),
                             'duration':duration,'gpus':g})
    batches=[];daily=[]
    for j,(key,jobs) in enumerate(buckets.items()):
        jobs=sorted(jobs,key=lambda x:(x['submit'],x['id']))
        daily.append({'date':key,'eligible':len(jobs),'selected':8 if len(jobs)>=8 else 0})
        if len(jobs)>=8:batches.append({'date':key,'train':j<14,'jobs':jobs[:8]})
    tr=[b for b in batches if b['train']];assert tr
    energy=sorted(j['duration']*j['gpus'] for b in tr for j in b['jobs'])
    peaks=sorted(int(sum(hourly(j['submit'],j['duration'],j['gpus']) for j in b['jobs']).max()) for b in tr)
    twice=lambda x:x[(len(x)-1)//2]+x[len(x)//2]
    d={'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'counts':counts,
       'daily':daily,'scale_E2':twice(energy),'capacity_C8':3*twice(peaks),
       'training_energy_gpu_seconds':energy,'training_peak_gpu_seconds':peaks,'batches':batches}
    (P/'trace_inputs.json').write_text(json.dumps(d,indent=2));print('FROZEN',counts,d['scale_E2'],d['capacity_C8'],flush=True)

def analyze():
    d=json.loads((P/'trace_inputs.json').read_text());E2=d['scale_E2'];C8=d['capacity_C8']
    M=np.array(list(product(range(4),repeat=8)),dtype=np.int8);AD=M>0;MASK=AD@(1<<np.arange(8))
    costs=((5*lcg(20260922,1024))//2**32).astype(np.int64).reshape(128,8)
    vals=(6+(19*lcg(20260929,1024))//2**32).astype(np.int64).reshape(128,8)
    rows=[];physical=[];minp=0;minu=0;balance=0
    choose=lambda ids,w:int(ids[np.argmax(w[ids])])
    def payment(j,ids,w,v,c,E):
        own=E2*v*AD[j]-c*E*np.maximum(M[j]-1,0)
        return np.array([w[choose(ids[M[ids,i]==0],w)]-w[j]+own[i] for i in range(8)])
    for b in d['batches']:
        if b['train']:continue
        a=np.zeros((8,4,36),dtype=np.int64);E=np.array([j['duration']*j['gpus'] for j in b['jobs']])
        for i,j in enumerate(b['jobs']):
            for m,h in enumerate([0,21600,43200],1):
                a[i,m]=hourly(j['submit'],j['duration'],j['gpus'],h);assert a[i,m].sum()==E[i]
        load=sum(a[i,M[:,i]] for i in range(8))
        good=np.flatnonzero(np.all(8*load<=C8,axis=1));half=good[np.all(M[good]<3,axis=1)]
        firm=good[np.all(M[good]<2,axis=1)];blind=np.flatnonzero(np.all(8*load[:,:12]<=C8,axis=1))
        physical.append({'date':b['date'],'full_feasible':len(good),'half_feasible':len(half),
                         'firm_feasible':len(firm),'blind_feasible':len(blind),'energy_gpu_seconds':E.tolist()})
        Q=np.maximum(M-1,0)*E
        for r,(v,c) in enumerate(zip(vals,costs)):
            w=E2*(AD@v)-Q@c
            jf,jh,jn,jb=[choose(ids,w) for ids in [good,half,firm,blind]]
            g=np.full(256,np.iinfo(np.int64).min,dtype=np.int64);np.maximum.at(g,MASK[good],w[good])
            seq=[]
            for order in [range(8),sorted(range(8),key=lambda i:(c[i],i)),sorted(range(8),key=lambda i:(-v[i],i))]:
                kept=0
                for i in order:
                    trial=kept|(1<<i)
                    if g[trial]>=g[kept]:kept=trial
                seq.append(choose(good[MASK[good]==kept],w))
            pf=payment(jf,good,w,v,c,E);ph=payment(jh,half,w,v,c,E);menus=[]
            for ids in [good,half]:
                menu=[]
                for i in range(8):
                    fixed=ids[M[ids,i]==1];absent=ids[M[ids,i]==0]
                    menu.append(None if not len(fixed) else int(w[choose(absent,w)]-w[choose(fixed,w)]+E2*v[i]))
                menus.append(menu)
            own=E2*v*AD[jf]-c*E*np.maximum(M[jf]-1,0);u=own-pf
            minp=min(minp,int(pf.min()));minu=min(minu,int(u.min()));balance=max(balance,abs(int(w[jf]-pf.sum()-u.sum())))
            js=[jf,jh,jn,*seq,jb]
            rows.append({'date':b['date'],'r':r,'w_num':[int(w[j]) for j in js],
              'modes':[M[j].tolist() for j in js],'payment_full_num':pf.tolist(),'payment_half_num':ph.tolist(),
              'firm_menu_full_num':menus[0],'firm_menu_half_num':menus[1],'firm_menu_change':int(menus[0]!=menus[1]),
              'admission_change_half':int(np.any(AD[jf]!=AD[jh])),
              'payment_change_half':int(np.any(pf!=ph)),'mode_change_half':int(np.any(M[jf]!=M[jh])),
              'blind_violation':int(np.any(8*load[jb]>C8)),
              'blind_excess_C8':int(max(0,(8*load[jb]-C8).max()))})
        print('HOLDOUT',b['date'],len(good),flush=True)
    audit={'min_payment_numerator':minp,'min_utility_numerator':minu,'max_accounting_error_numerator':balance}
    assert minp==minu==balance==0
    result={'input_sha256':hashlib.sha256((P/'trace_inputs.json').read_bytes()).hexdigest(),
      'objective_denominator':E2,'policy_order':['full','half','firm','arrival','cost','value','blind'],
      'physical':physical,'records':rows,'audit':audit}
    (P/'trace_results.json').write_text(json.dumps(result,separators=(',',':')))
    w=np.array([r['w_num'][:6] for r in rows])/E2;ok=w[:,0]>0;loss=100*(w[ok,0,None]-w[ok])/w[ok,0,None]
    summary={'cases':len(rows),'days':len(physical),'zero_optimum_cases':int((~ok).sum()),
      'mean_welfare':w.mean(axis=0).tolist(),'mean_loss_percent':loss.mean(axis=0).tolist(),
      'max_loss_percent':loss.max(axis=0).tolist(),
      **{k:int(sum(x[k] for x in rows)) for k in ['admission_change_half','payment_change_half','mode_change_half','blind_violation','firm_menu_change']},'audit':audit}
    (P/'trace_summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='extract':extract(sys.argv[2])
    else:analyze()
