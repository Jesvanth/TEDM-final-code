import argparse,glob,json,matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
p=argparse.ArgumentParser();p.add_argument('--root',default='outputs');p.add_argument('--out',default='results/roc.png');a=p.parse_args();plt.figure()
for model in ['text_bert','misa','mmbt','early_fusion','tedm']:
 ys=[];ps=[]
 for f in glob.glob(f'{a.root}/{model}/seed_*/metrics.json'):
  d=json.load(open(f));ys+=d['y_true'];ps+=d['y_prob']
 if ys:
  x,y,_=roc_curve(ys,ps);plt.plot(x,y,label=f'{model} AUC={auc(x,y):.3f}')
plt.plot([0,1],[0,1],'--');plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');plt.legend();plt.tight_layout();plt.savefig(a.out,dpi=200)
