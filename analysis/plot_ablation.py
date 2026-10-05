import argparse,glob,json,pandas as pd,matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/ablations');p.add_argument('--out',default='results/ablation.png');a=p.parse_args();r=[]
for f in glob.glob(a.root+'/*/seed_*/metrics.json'):
 d=json.load(open(f))['metrics'];r.append({'variant':d['ablation'],'f1':d['f1']})
s=pd.DataFrame(r).groupby('variant').f1.mean().sort_values();plt.figure();plt.bar(s.index,s.values);plt.ylabel('Mean F1 (%)');plt.xticks(rotation=25,ha='right');plt.tight_layout();plt.savefig(a.out,dpi=200)
