import argparse,pandas as pd
from sklearn.model_selection import train_test_split
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out-prefix',required=True); p.add_argument('--seed',type=int,default=2026); a=p.parse_args()
df=pd.read_csv(a.manifest); ids=df[['dialogue_id','label']].drop_duplicates('dialogue_id')
tr,tmp=train_test_split(ids,test_size=.2,random_state=a.seed,stratify=ids.label)
va,te=train_test_split(tmp,test_size=.5,random_state=a.seed,stratify=tmp.label)
for n,x in [('train',tr),('val',va),('test',te)]: x[['dialogue_id']].to_csv(f'{a.out_prefix}_{n}.txt',index=False,header=False)
