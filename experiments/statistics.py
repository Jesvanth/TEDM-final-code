import argparse,json,glob,pandas as pd
from scipy.stats import ttest_rel
p=argparse.ArgumentParser();p.add_argument('--root',default='outputs');p.add_argument('--out',default='results/statistics.csv');a=p.parse_args();rows=[]
for f in glob.glob(a.root+'/*/seed_*/metrics.json'):
 d=json.load(open(f))['metrics'];rows.append(d)
df=pd.DataFrame(rows); ted=df[df.model=='tedm'].sort_values('seed')
out=[]
for name in ['text_bert','misa','mmbt','early_fusion']:
 b=df[df.model==name].sort_values('seed');m=ted[['seed','f1']].merge(b[['seed','f1']],on='seed',suffixes=('_tedm','_base'));t,pv=ttest_rel(m.f1_tedm,m.f1_base);out.append({'comparison':f'TEDM vs {name}','mean_difference_pp':(m.f1_tedm-m.f1_base).mean(),'t_statistic':t,'p_value':pv})
pd.DataFrame(out).to_csv(a.out,index=False)
