import argparse,glob,json,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',default='outputs');p.add_argument('--out',default='results/table5_five_seed_summary.csv');a=p.parse_args();rows=[]
for f in glob.glob(a.root+'/*/seed_*/metrics.json'):
 d=json.load(open(f))['metrics']; rows.append(d)
df=pd.DataFrame(rows); cols=['accuracy','precision','recall','f1','auc'];s=df.groupby('model')[cols].agg(['mean','std']);s.to_csv(a.out);print(s)
