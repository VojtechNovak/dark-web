from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(a.results/'ranking_all.csv').query("stratum=='all'")
names=['preferential_attachment','multimodal_logreg','gae_full','vgae_full'];labels=['Preferential\nattachment','Multimodal\nLR','GAE','VGAE']
colors=['#85878a','#5d91bd','#4b9b8b','#806ab1'];x=np.arange(4)
fig,ax=plt.subplots(1,2,figsize=(8.0,3.2))
for b,col,title in zip(ax,['mrr','hits10'],['MRR','Hits@10']):
 s=d.groupby('model')[col].agg(['mean','std']).loc[names]
 b.bar(x,s['mean'],yerr=s['std'],color=colors,capsize=3,width=.62)
 b.set_xticks(x,labels,fontsize=8);b.set_ylabel(title);b.spines[['top','right']].set_visible(False);b.grid(axis='y',alpha=.15);b.set_axisbelow(True)
fig.tight_layout();fig.savefig(a.out/'filtered_ranking.png',dpi=300,bbox_inches='tight');plt.close(fig)
