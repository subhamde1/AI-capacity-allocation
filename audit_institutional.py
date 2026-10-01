"""Separate exact verification using per-second binning and direct mode enumeration."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
P=Path(__file__).resolve().parent
recovery="--recovery" in sys.argv;prefix="recovery_" if recovery else ""
with np.load(P/(prefix+"institutional_records.npz")) as raw:rec={k:raw[k] for k in raw.files}
proto=json.loads((P/"field_grounding_protocol.json").read_text())
batches=[b for b in json.loads((P/"trace_inputs.json").read_text())["batches"] if not b["train"]]
# Base-four enumeration is independent of the primary program construction.
M=((np.arange(65536)[:,None]//(4**np.arange(7,-1,-1)))%4).astype(np.int64)
admit=M>0;delay=np.maximum(M-1,0);masks=admit@(1<<np.arange(8))
def sequence(seed):
 out=[]
 for _ in range(1024):
  seed=(1664525*seed+1013904223)%(2**32);out.append(seed)
 return np.array(out,dtype=np.int64).reshape(128,8)
values=6+19*sequence(20260929)//2**32;costs=5*sequence(20260922)//2**32
count=0;direct=0;endpoints=0
for di,b in enumerate(batches):
 profile=np.zeros((8,4,36),dtype=np.int64)
 E=[]
 for i,j in enumerate(b["jobs"]):
  E.append(j["duration"]*j["gpus"])
  for mode,offset in [(1,0),(2,21600),(3,43200)]:
   seconds=np.arange(j["submit"]+offset,j["submit"]+offset+j["duration"])
   profile[i,mode]=j["gpus"]*np.bincount(seconds//3600,minlength=36)
 E=np.array(E);load=sum(profile[i,M[:,i]] for i in range(8))
 for ri,R in enumerate([2,3,4,6]):
  for zi,Z in enumerate([1,2,4] if recovery else [1,2,3]):
   gamma=3 if recovery else Z;D=Z if recovery else 2
   capacity=R*8489*np.where(np.arange(36)%24<6,gamma,1)*np.where(np.arange(36)<12,2,D)
   full=np.all(load*16<=capacity,axis=1)
   half=full&np.all(M!=3,axis=1)
   blind=np.all(load[:,:12]*16<=capacity[:12],axis=1)
   for ai,A in enumerate([1,2,4,8,16]):
    for bi,B in enumerate([0,1,2,4,8,16]):
     start=((((di*4+ri)*3+zi)*5+ai)*6+bi)*128
     rows=slice(start,start+128);ids=rec["allocation_idx"][rows]
     for policy,allowed in enumerate([full,half,blind,full]):
      assert np.all(allowed[ids[:,policy]])
      modes=M[ids[:,policy]]
      own=156*A*values*(modes>0)-B*costs*E*np.maximum(modes-1,0)
      assert np.array_equal(own.sum(axis=1),rec["welfare_num"][rows,policy]);count+=128
     modes=M[ids[:,0]];own=156*A*values*(modes>0)-B*costs*E*np.maximum(modes-1,0)
     pay=rec["payments_num"][rows]
     assert np.all(pay>=0) and np.all(own-pay>=0) and np.all(pay[modes==0]==0)
     if (A,B) not in [(1,0),(1,16),(4,4),(16,1)]:continue
     for v in [0,127]:
      n=start+v
      objective=156*A*(admit@values[v])-B*(delay@(costs[v]*E))
      def pick(allowed):return int(np.argmax(np.where(allowed,objective,-10**14)))
      for policy,allowed in enumerate([full,half,blind]):
       j=pick(allowed)
       assert j==rec["allocation_idx"][n,policy]
       assert objective[j]==rec["welfare_num"][n,policy]
      kept=0;candidate=0
      for i in sorted(range(8),key=lambda i:(-values[v,i],i)):
       allowed=full&(masks==(kept|(1<<i)))
       if allowed.any():
        trial=pick(allowed)
        if objective[trial]>=objective[candidate]:kept|=1<<i;candidate=trial
      assert candidate==rec["allocation_idx"][n,3]
      for policy,allowed in enumerate([full,half]):
       for i in range(8):
        excluded=int(objective[pick(allowed&(M[:,i]==0))])
        fixed=allowed&(M[:,i]==1)
        menu=-1 if not fixed.any() else excluded-int(objective[pick(fixed)])+156*A*int(values[v,i])
        assert menu==rec["firm_full_num" if policy==0 else "firm_half_num"][n,i];endpoints+=1
        if policy==0:
         j=int(rec["allocation_idx"][n,0]);mode=int(M[j,i])
         own=156*A*int(values[v,i])*(mode>0)-B*int(costs[v,i]*E[i])*max(mode-1,0)
         assert excluded-int(objective[j])+own==rec["payments_num"][n,i]
      direct+=1
result={"recorded_cases":len(rec["welfare_num"]),"policy_primal_records_checked":count,"direct_enumeration_cases":direct,"sampled_fixed_profile_menu_endpoints_checked":endpoints,"errors":0,"scope":"Independent formulation from second-by-second binning and direct 65536-mode enumeration; same public inputs and exact integer definitions. Author-side computational verification.","script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert (count,direct,endpoints)==(2580480,1344,21504)
(P/(prefix+"institutional_audit.json")).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
