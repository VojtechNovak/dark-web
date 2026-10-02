#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--csv', default='pa_degree_diagnostics_20seeds.csv')
p.add_argument('--out', default='degree_bias_audit.png')
a=p.parse_args()
df=pd.read_csv(a.csv)
order=['random','degree_corrected','source_degree_matched']
data=[df.loc[df.regime==r,'pa_auc'].to_numpy() for r in order]
fig,ax=plt.subplots(figsize=(7.2,4.6))
ax.boxplot(data, tick_labels=['Random','Degree-corrected','Source + target-degree matched'], showmeans=True)
ax.axhline(0.5, linestyle='--', linewidth=1)
ax.set_ylabel('Preferential-attachment ROC-AUC')
ax.set_xlabel('Negative-sampling protocol')
ax.set_ylim(0.45,1.02)
ax.set_title('Degree shortcut collapses under degree-aware evaluation (20 splits)')
fig.tight_layout()
fig.savefig(a.out,dpi=220,bbox_inches='tight')
print(Path(a.out).resolve())
