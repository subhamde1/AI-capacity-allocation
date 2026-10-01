"""Verify the broad/narrow family using actual integer load matrices."""
from pathlib import Path
import json,itertools
import numpy as np
P=Path(__file__).resolve().parent
records=json.loads((P/"bundle_sensitivity.json").read_text());count=0
for r in [2,3,4,8]:
 # All loads and capacities are multiplied by r: every applicant uses r units.
 loads=np.zeros((r+1,r+1),dtype=np.int64);loads[0,:r]=1
 for j in range(1,r+1):loads[j,j-1]=1;loads[j,r]=r-1
 cap=np.array([1]*r+[r*(r-1)]);assert np.all(loads.sum(axis=1)==r)
 portfolios=np.array(list(itertools.product([0,1],repeat=r+1)))
 feasible=np.all(portfolios@loads<=cap,axis=1)
 masks=portfolios@(1<<np.arange(r+1));family=set(masks[feasible].tolist())
 assert family=={0,1}|{2*s for s in range(1,2**r)}
 for a in [z for z in records if z["r"]==r]:
  k=a["narrow_value"];z=a["broad_ratio_num"];h=a["cost_ratio_num"]
  nets=np.array([(z-h)*k]+[(4-h)*k]*r)
  w=portfolios@nets;assert max(w[feasible])==a["optimum_num4"]
  gm=a["greedy_mask"];assert gm in family
  assert sum(nets[i] for i in range(r+1) if gm&(1<<i))==a["greedy_num4"]
  count+=1
out={"cases":count,"errors":0,"common_capacity":"r-1, equal to 2 only when r=3","method":"Explicit integer load matrices, all binary portfolios, energy conservation and exact objective enumeration."}
assert count==512
(P/"bundle_physical_audit.json").write_text(json.dumps(out,indent=2));print(out)
