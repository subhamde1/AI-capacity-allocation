
from itertools import product
from fractions import Fraction
import json
from pathlib import Path
out={}
# All allocations of the simplified auction in Proposition 2.
modes=list(product(range(3),repeat=3)) # reject, firm, defer
physical=lambda m:sum(x==1 for x in m)<=1

def best(c,removed=None):
 cand=[m for m in modes if physical(m) and (removed is None or m[removed]==0)]
 w=lambda m:sum(0 if x==0 else 10-(c[i] if x==2 else 0) for i,x in enumerate(m))
 m=max(cand,key=w);return w(m),m
for reports in [[3,2,1],[3,2,4]]:
 w,m=best(reports);p=[]
 for i in range(3):
  own=0 if m[i]==0 else 10-(reports[i] if m[i]==2 else 0)
  p.append(best(reports,i)[0]-(w-own))
 retained=[i for i in range(3) if m[i]==1]
 deferred=[i for i in range(3) if m[i]==2]
 if deferred:retained.append(min(deferred))
 utilities=[0 if i not in retained else 10-([3,2,1][i] if m[i]==2 else 0)-p[i] for i in range(3)]
 out[str(reports)]={'W':w,'modes':m,'payments_before_repair':p,
   'retained_one_based':[i+1 for i in retained],'true_utilities_after_repair':utilities}
# Every subset of the broad/narrow construction for r=2,...,8.
for r in range(2,9):
 a=[[Fraction(1,r)]*r+[0]]
 for i in range(r):a.append([Fraction(int(j==i),r) for j in range(r)]+[Fraction(r-1,r)])
 C=[Fraction(1,r)]*r+[r-1]
 feasible=[]
 for x in product([0,1],repeat=r+1):
  if all(sum(x[i]*a[i][t] for i in range(r+1))<=C[t] for t in range(r+1)):feasible.append(x)
 assert max(map(sum,feasible))==r
 assert max(sum(x) for x in feasible if x[0])==1
 assert all(sum(row)==1 for row in a)
out['ranking_r_values']=list(range(2,9));out['repair_gain']=7
(Path(__file__).resolve().parent/'arithmetic_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
