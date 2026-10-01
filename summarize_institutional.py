from pathlib import Path
from itertools import product
import json,csv
import numpy as np
P=Path(__file__).resolve().parent
R=[2,3,4,6];G=[1,2,3];A=[1,2,4,8,16];B=[0,1,2,4,8,16]
with np.load(P/"institutional_records.npz") as z:x={k:z[k] for k in z.files}
with np.load(P/"recovery_institutional_records.npz") as z:xr={k:z[k] for k in z.files}
f=x["flags"].reshape(14,4,3,5,6,128,7);cell=f.sum(axis=(0,5));detail=[]
for ri,gi,ai,bi in product(range(4),range(3),range(5),range(6)):
 detail.append({"rho":R[ri]/4,"gamma":G[gi],"alpha":A[ai]/4,"beta":B[bi]/4,**{k:int(z) for k,z in zip(["blind_infeasible","blind_admission","catalog_admission","catalog_menu","value_reversal","priority_loss","zero_optimum"],cell[ri,gi,ai,bi])}})
def csvwrite(name,rows):
 with (P/name).open("w",newline="") as out:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
csvwrite("institutional_complete_grid.csv",detail)
summary={"cases":645120,"cells":360,"cases_per_cell":1792,"flag_order":json.loads((P/"institutional_manifest.json").read_text())["flag_order"],"counts":f.sum(axis=(0,1,2,3,4,5)).tolist(),"cell_min_counts":cell.min(axis=(0,1,2,3)).tolist(),"cell_max_counts":cell.max(axis=(0,1,2,3)).tolist(),"cells_with_event":(cell>0).sum(axis=(0,1,2,3)).tolist(),"published_profile_cases":215040,"published_profile_counts":f[:,:,2].sum(axis=(0,1,2,3,4)).tolist(),"exchange_failure_settings":sum(z["exchange_witness"] is not None for z in json.loads((P/"institutional_physical.json").read_text()))}
m=xr["firm_full_num"].reshape(14,4,3,5,6,128,8)
ids=xr["allocation_idx"][:,0].reshape(14,4,3,5,6,128)
mode=lambda j:(j[...,None]//(4**np.arange(7,-1,-1)))%4
pairs=[];pair_rows=[]
for k in [0,1]:
 common=(m[:,:,k]>=0)&(m[:,:,k+1]>=0)
 change=common&(m[:,:,k]!=m[:,:,k+1]);up=common&(m[:,:,k+1]>m[:,:,k]);down=common&(m[:,:,k+1]<m[:,:,k])
 adm=np.any((mode(ids[:,:,k])>0)!=(mode(ids[:,:,k+1])>0),axis=-1);price=change.any(axis=-1)
 pairs.append({"from_delta":[.5,1][k],"to_delta":[1,2][k],"comparisons":215040,"admission_change_cases":int(adm.sum()),"common_finite_menu_change_cases":int(price.sum()),"cases_with_common_finite_menu":int(common.any(axis=-1).sum()),"common_profile_comparisons":int(common.sum()),"profile_price_increases":int(up.sum()),"profile_price_decreases":int(down.sum()),"cells_with_admission_change":int((adm.sum(axis=(0,4))>0).sum()),"cells_with_menu_change":int((price.sum(axis=(0,4))>0).sum())})
 ca=adm.sum(axis=(0,4));cp=price.sum(axis=(0,4))
 for ri,ai,bi in product(range(4),range(5),range(6)):
  pair_rows.append({"rho":R[ri]/4,"alpha":A[ai]/4,"beta":B[bi]/4,"delta_from":[.5,1][k],"delta_to":[1,2][k],"admission_changes":int(ca[ri,ai,bi]),"common_menu_changes":int(cp[ri,ai,bi]),"comparisons":1792})
csvwrite("recovery_complete_grid.csv",pair_rows);summary["recovery_pairs"]=pairs
for name in x:
 tail=x[name].shape[1:]
 lhs=x[name].reshape((14,4,3,5,6,128)+tail)[:,:,2]
 rhs=xr[name].reshape((14,4,3,5,6,128)+tail)[:,:,1]
 assert np.array_equal(lhs,rhs),name
summary["recovery_anchor_matches"]=215040
gm=x["firm_full_num"].reshape(14,4,3,5,6,128,8);gi=x["allocation_idx"][:,0].reshape(14,4,3,5,6,128);gp=[]
for k in [0,1]:
 common=(gm[:,:,k]>=0)&(gm[:,:,k+1]>=0);change=common&(gm[:,:,k]!=gm[:,:,k+1])
 adm=np.any((mode(gi[:,:,k])>0)!=(mode(gi[:,:,k+1])>0),axis=-1)
 gp.append({"from_gamma":k+1,"to_gamma":k+2,"comparisons":215040,"admission_change_cases":int(adm.sum()),"common_finite_menu_change_cases":int(change.any(axis=-1).sum()),"common_profile_comparisons":int(common.sum()),"newly_feasible_reference_profiles":int(((gm[:,:,k]<0)&(gm[:,:,k+1]>=0)).sum())})
summary["overnight_shape_pairs"]=gp
bundle=json.loads((P/"bundle_sensitivity.json").read_text())
summary["bundle"]={"cases":len(bundle),"higher_value_exclusions":sum(z["strict_higher_value_excluded"] for z in bundle),"priority_losses":sum(z["greedy_loss"]>0 for z in bundle),"positive_optimum_cases":sum(z["optimum_num4"]>0 for z in bundle),"min_greedy_ratio":min(z["greedy_num4"]/z["optimum_num4"] for z in bundle if z["optimum_num4"]>0),"by_r":[{"r":r,"cases":sum(z["r"]==r for z in bundle),"higher_value_exclusions":sum(z["r"]==r and z["strict_higher_value_excluded"] for z in bundle),"priority_losses":sum(z["r"]==r and z["greedy_loss"]>0 for z in bundle)} for r in [2,3,4,8]]}
# Check the complete recorded aggregate results.
assert summary["counts"]==[207305,206621,81182,72757,306487,10926,7680]
assert summary["cell_min_counts"]==[60,60,27,13,108,0,0]
assert summary["cell_max_counts"]==[1536,1536,1152,1143,1411,252,128]
assert summary["published_profile_counts"]==[59424,58873,14633,14404,69621,2244,0]
assert [(p["admission_change_cases"],p["common_finite_menu_change_cases"],p["profile_price_increases"],p["profile_price_decreases"]) for p in pairs]==[(13287,13770,2643,65584),(51477,41460,2055,149950)]
assert [(p["admission_change_cases"],p["common_finite_menu_change_cases"]) for p in gp]==[(155540,102029),(97521,109098)]
assert (summary["bundle"]["higher_value_exclusions"],summary["bundle"]["priority_losses"],summary["bundle"]["min_greedy_ratio"])==(164,212,.125)
(P/"institutional_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
