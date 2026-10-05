import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix

def metrics(y,p,threshold=.5):
    y=np.asarray(y);p=np.asarray(p);h=(p>=threshold).astype(int)
    return {'accuracy':accuracy_score(y,h)*100,'precision':precision_score(y,h,zero_division=0)*100,'recall':recall_score(y,h,zero_division=0)*100,'f1':f1_score(y,h,zero_division=0)*100,'auc':roc_auc_score(y,p) if len(set(y))>1 else float('nan'),'confusion_matrix':confusion_matrix(y,h).tolist()}
