import torch,pandas as pd
from torch.utils.data import Dataset
class DialogueFeatures(Dataset):
    def __init__(self,manifest,ids_file=None):
        d=pd.read_csv(manifest)
        if ids_file:
            ids=set(x.strip() for x in open(ids_file) if x.strip()); d=d[d.dialogue_id.astype(str).isin(ids)]
        self.rows=d.to_dict('records')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; z=torch.load(r['feature_path'],map_location='cpu'); return z

def collate(batch):
    B=len(batch); L=max(x['text'].shape[0] for x in batch); td=batch[0]['text'].shape[1]; ad=batch[0]['audio'].shape[1]; vd=batch[0]['video'].shape[1]
    T=torch.zeros(B,L,td);A=torch.zeros(B,L,ad);V=torch.zeros(B,L,vd);M=torch.zeros(B,L,dtype=torch.bool);Y=torch.zeros(B)
    for i,x in enumerate(batch):
        n=x['text'].shape[0]; T[i,:n]=x['text'];A[i,:n]=x['audio'];V[i,:n]=x['video'];M[i,:n]=1;Y[i]=x['labels'][-1].float()
    return T,A,V,M,Y
