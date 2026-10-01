
"""Run from any clean working directory; uses only the adjacent source files."""
import runpy,json,sys,platform,hashlib
from pathlib import Path
import numpy as np
import scipy
P=Path(__file__).resolve().parent
x=runpy.run_path(str(P/'baseline_code.py'))
a=np.array(x['all_rows']);loss=np.array(x['all_losses'])
assert a.shape==(3456,17)
assert x['checks']==51840 and x['max_gain']==0
assert x['min_p']==0 and x['min_u']==0
assert abs(a[:,0].mean()-88.73813657407408)<1e-9
assert np.max(abs(a[:,0]-a[:,11]-a[:,12]))<1e-9
summary={'mean':a.mean(axis=0).tolist(),'max_losses':loss.max(axis=0).tolist(),
 'audit':{'reports':x['checks'],'max_gain':x['max_gain'],
 'min_payment':x['min_p'],'min_utility':x['min_u'],'milp_difference':x['max_gap']},
 'by_kappa':{str(k):np.array(v).mean(axis=0).tolist() for k,v in x['by_kappa'].items()}}
(P/'baseline_results.json').write_text(json.dumps(summary,indent=2))
# Preserve all finite-design records for figure and table generation.
np.save(P/'baseline_rows.npy',a)
runpy.run_path(str(P/'check_arithmetic.py'),run_name='__main__')
sys.path.insert(0,str(P))
import extended_experiments as e
e.structural_audit();e.economic()
# The time-limited scale run is separate: python extended_experiments.py scale.
print('PASS: original design, accounting, report checks, structural and economic extensions')
