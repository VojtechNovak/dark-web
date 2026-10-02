#!/usr/bin/env python3
"""Run the completed follow-up protocol, controls, verification and ranking plot."""
import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('reproduced_checks'))
p.add_argument('--supplement',type=Path,default=Path(__file__).resolve().parents[1]/'reference/OnlineResource1')
a=p.parse_args();code=Path(__file__).resolve().parent;env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for script in ['run_checks.py','compression_budget.py','verify_checks.py']:
 subprocess.run([sys.executable,str(code/script),'--supplement',str(a.supplement),'--out',str(a.out)],env=env,check=True)
subprocess.run([sys.executable,str(code/'plot_ranking.py'),'--results',str(a.out),'--out',str(a.out/'figures')],env=env,check=True)
