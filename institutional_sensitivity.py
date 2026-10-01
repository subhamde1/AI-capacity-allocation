"""Exact institutionally grounded sensitivity. Run beside the frozen protocol and trace inputs.
No input is estimated from holdout outcomes. All money is recorded over denominator 624.
"""
from pathlib import Path
from itertools import product
import json,hashlib,time,platform,sys
import numpy as np
P=Path(__file__).resolve().parent
INF=np.int64(10**14);BAD=65536
M=np.array(list(product(range(4),repeat=8)),dtype=np.int8)
AD=M>0;MASK=AD@(1<<np.arange(8));BITS=((np.arange(256)[:,None]>>np.arange(8))&1).astype(np.int64)
def lcg(seed,count):
 z=seed;out=[]
 for _ in range(count):z=(1664525*z+1013904223)%2**32;out.append(z)
 return np.array(out,dtype=np.int64)
def profiles(jobs):
 out=np.zeros((8,4,36),dtype=np.int64)
 for i,j in enumerate(jobs):
  s,d,g=j["submit"],j["duration"],j["gpus"]
  for m,h in enumerate([0,21600,43200],1):
   for t in range(36):out[i,m,t]=g*max(0,min(3600*(t+1),s+h+d)-max(3600*t,s+h))
  assert np.all(out[i,1:].sum(axis=1)==d*g)
 return out
def compress(ids,allcost):
 phi=np.full((128,256),INF,dtype=np.int64)
 arg=np.full((128,256),BAD,dtype=np.int32);first=np.full(256,BAD,dtype=np.int32)
 mask=MASK[ids]
 for s in np.unique(mask):
  js=ids[mask==s];x=allcost[js];k=np.argmin(x,axis=0)
  phi[:,s]=x[k,np.arange(128)];arg[:,s]=js[k];first[s]=js[0]
 return phi,arg,first
def scores(comp,sv,A,B):
 phi,arg,first=comp
 w=156*A*sv-B*phi;w[:,first==BAD]=-INF
 return w,(np.broadcast_to(first,arg.shape) if B==0 else arg)
def select(w,arg):
 best=w.max(axis=1);ids=np.where(w==best[:,None],arg,BAD).min(axis=1)
 assert np.all(ids<BAD)
 return best,ids
def exchange_witness(family):
 fs=np.flatnonzero(family).tolist()
 for a in fs:
  for b in fs:
   if a.bit_count()>=b.bit_count():continue
   remaining=b&~a
   if not any(family[a|(1<<i)] for i in range(8) if remaining&(1<<i)):
    return {"small_mask":a,"large_mask":b,"small":[i+1 for i in range(8) if a&(1<<i)],"large":[i+1 for i in range(8) if b&(1<<i)]}
 return None
def bundle_diagnostic(cfg):
 records=[]
 for r,k,z,h in product(cfg["r"],cfg["narrow_value"],cfg["broad_value_ratio_num_over_4"],cfg["common_exposure_cost_ratio_num_over_4"]):
  narrow=4*k;broad=z*k;cost=h*k
  nets=np.array([broad-cost]+[narrow-cost]*r,dtype=np.int64)
  families=[0,1]+[2*s for s in range(1,2**r)]
  w=np.array([sum(nets[i] for i in range(r+1) if s&(1<<i)) for s in families])
  optimum=int(w.max());accepted=families[int(np.argmax(w))];current=0
  for i in sorted(range(r+1),key=lambda i:(-([broad]+[narrow]*r)[i],i)):
   trial=current|(1<<i)
   if trial in families and sum(nets[j] for j in range(r+1) if trial&(1<<j))>=sum(nets[j] for j in range(r+1) if current&(1<<j)):current=trial
  greedy=sum(nets[j] for j in range(r+1) if current&(1<<j))
  assert optimum==max(0,broad-cost,r*max(0,narrow-cost))
  records.append({"r":r,"narrow_value":k,"broad_ratio_num":z,"cost_ratio_num":h,"optimum_num4":optimum,"greedy_num4":int(greedy),"optimal_mask":accepted,"greedy_mask":current,"strict_higher_value_excluded":bool(broad>narrow and accepted>1),"greedy_loss":int(optimum-greedy)})
 return records
def run():
 start=time.perf_counter();recovery="--recovery" in sys.argv;prefix="recovery_" if recovery else ""
 protocol=json.loads((P/"field_grounding_protocol.json").read_text())
 raw=(P/"trace_inputs.json").read_bytes();assert hashlib.sha256(raw).hexdigest()==protocol["source_input_sha256"]
 d=json.loads(raw);batches=[b for b in d["batches"] if not b["train"]]
 vals=(6+(19*lcg(protocol["value_seed"],1024))//2**32).reshape(128,8)
 costs=((5*lcg(protocol["cost_seed"],1024))//2**32).reshape(128,8);sv=vals@BITS.T
 original=json.loads((P/"trace_original_results.json").read_text())["records"]
 original={(r["date"],r["r"]):r for r in original}
 physical=[];cellstats=[];allw=[];allids=[];allp=[];allmf=[];allmh=[];allflags=[];baseline=0
 for di,b in enumerate(batches):
  a=profiles(b["jobs"]);E=np.array([j["duration"]*j["gpus"] for j in b["jobs"]])
  load=sum(a[i,M[:,i]] for i in range(8));Q=np.maximum(M-1,0)*E;allcost=Q@costs.T
  for R,Z in product(protocol["capacity_scale_R_over_4"],[1,2,4] if recovery else protocol["overnight_ratio_G"]):
   G=3 if recovery else Z;D=Z if recovery else 2
   cap=R*8489*np.where(np.arange(36)%24<6,G,1)*np.where(np.arange(36)>=12,D,2)
   feasible=np.all(16*load<=cap,axis=1)
   full=np.flatnonzero(feasible);half=full[np.all(M[full]<3,axis=1)]
   blind=np.flatnonzero(np.all(16*load[:,:12]<=cap[:12],axis=1))
   comps=[compress(ids,allcost) for ids in [full,half,blind]]
   fixed=[[compress(ids[M[ids,i]==1],allcost) for i in range(8)] for ids in [full,half]]
   witness=exchange_witness(comps[0][2]<BAD)
   physical.append({"date":b["date"],"R":R,"G":G,"D":D,"feasible_modes":len(full),"feasible_sets":int((comps[0][2]<BAD).sum()),"exchange_witness":witness})
   for A,B in product(protocol["value_scale_A_over_4"],protocol["cost_scale_B_over_4"]):
    ws=[];js=[];sc=[];args=[]
    for c in comps:
     w,arg=scores(c,sv,A,B);best,j=select(w,arg);ws.append(best);js.append(j);sc.append(w);args.append(arg)
    kept=np.zeros(128,dtype=np.int64);rr=np.arange(128);order=np.argsort(-vals,axis=1,kind="stable")
    for k in range(8):
     trial=kept|(1<<order[:,k]);kept=np.where(sc[0][rr,trial]>=sc[0][rr,kept],trial,kept)
    jg=args[0][rr,kept];wg=sc[0][rr,kept];ws.append(wg);js.append(jg)
    w=np.stack(ws,axis=1);idx=np.stack(js,axis=1)
    assert np.all(w[:,0]>=w[:,1]) and np.all(w[:,2]>=w[:,0]) and np.all(w[:,0]>=wg)
    absent=[];menus=[]
    for c,which in enumerate([full,half]):
     wc=sc[c];exc=np.stack([wc[:,BITS[:,i]==0].max(axis=1) for i in range(8)],axis=1);absent.append(exc);mm=[]
     for i in range(8):
      wf,_=scores(fixed[c][i],sv,A,B);best=wf.max(axis=1)
      mm.append(np.where(best==-INF,-1,exc[:,i]-best+156*A*vals[:,i]))
     menus.append(np.stack(mm,axis=1));assert np.all(menus[-1]>=-1)
    modes=M[idx[:,0]];own=156*A*vals*(modes>0)-B*costs*E*np.maximum(modes-1,0)
    pay=absent[0]-w[:,0,None]+own;utility=own-pay
    assert np.all(pay>=0) and np.all(utility>=0)
    assert np.all(pay[modes==0]==0) and np.all(own.sum(axis=1)==w[:,0])
    assert np.all(pay.sum(axis=1)+utility.sum(axis=1)==w[:,0])
    blindbad=~feasible[idx[:,2]];assert np.array_equal(blindbad,idx[:,0]!=idx[:,2])
    rev=np.any((vals[:,:,None]>vals[:,None,:])&(~(modes>0))[:,:,None]&(modes>0)[:,None,:],axis=(1,2))
    flags=np.stack([blindbad,MASK[idx[:,0]]!=MASK[idx[:,2]],MASK[idx[:,0]]!=MASK[idx[:,1]],np.any(menus[0]!=menus[1],axis=1),rev,w[:,0]>wg,w[:,0]==0],axis=1)
    if not recovery and (R,G,A,B)==(3,1,4,4):
     for v in range(128):
      old=original[(b["date"],v)]
      assert w[v,:].tolist()==[4*old["w_num"][k] for k in [0,1,6,5]]
      assert [M[j].tolist() for j in idx[v]]==[old["modes"][k] for k in [0,1,6,5]]
      assert pay[v].tolist()==[4*x for x in old["payment_full_num"]]
      assert menus[0][v].tolist()==[-1 if x is None else 4*x for x in old["firm_menu_full_num"]]
      baseline+=1
    cellstats.append({"date":b["date"],"R":R,"G":G,"D":D,"A":A,"B":B,"cases":128,"counts":flags.sum(axis=0).tolist(),"welfare_sums_num":w.sum(axis=0).tolist(),"max_greedy_loss_num":int((w[:,0]-wg).max()),"max_menu_difference_num":int(np.max(np.abs(menus[0]-menus[1])))})
    allw.append(w);allids.append(idx);allp.append(pay);allmf.append(menus[0]);allmh.append(menus[1]);allflags.append(flags)
   print("PHYSICAL",b["date"],R,G,D,flush=True)
 arrays={"welfare_num":np.concatenate(allw),"allocation_idx":np.concatenate(allids),"payments_num":np.concatenate(allp),"firm_full_num":np.concatenate(allmf),"firm_half_num":np.concatenate(allmh),"flags":np.concatenate(allflags)}
 assert len(arrays["welfare_num"])==protocol["case_count"] and baseline==(0 if recovery else 1792)
 np.savez_compressed(P/(prefix+"institutional_records.npz"),**arrays)
 (P/(prefix+"institutional_cells.json")).write_text(json.dumps(cellstats,separators=(",",":")))
 (P/(prefix+"institutional_physical.json")).write_text(json.dumps(physical,indent=2))
 (P/(prefix+"bundle_sensitivity.json")).write_text(json.dumps(bundle_diagnostic(protocol["bundle_diagnostic"]),indent=2))
 manifest={"order":"date,R,D,A,B,financial_vector" if recovery else "date,R,G,A,B,financial_vector","dates":[b["date"] for b in batches],"R":protocol["capacity_scale_R_over_4"],"G":[3] if recovery else protocol["overnight_ratio_G"],"D":[1,2,4] if recovery else [2],"A":protocol["value_scale_A_over_4"],"B":protocol["cost_scale_B_over_4"],"objective_denominator":624,"policy_order":["full","half","blind","gross_value_order"],"flag_order":["blind_infeasible","blind_admission_change","half_admission_change","half_firm_menu_change","gross_value_reversal","gross_value_greedy_loss","zero_optimum"],"menu_missing":-1,"source_sha256":protocol["source_input_sha256"],"protocol_sha256":hashlib.sha256((P/"field_grounding_protocol.json").read_bytes()).hexdigest(),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"baseline_cases_matched":baseline,"elapsed_seconds":time.perf_counter()-start,"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"exact_checks":"Feasibility, objective ordering, pivot nonnegativity, truthful IR, zero charges for rejection, and welfare/payment/utility accounting."}
 if recovery:manifest["recovery_addendum_sha256"]=hashlib.sha256((P/"recovery_protocol_addendum.json").read_bytes()).hexdigest()
 (P/(prefix+"institutional_manifest.json")).write_text(json.dumps(manifest,indent=2));print("COMPLETE",manifest,flush=True)
if __name__=="__main__":run()
