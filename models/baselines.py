import torch, torch.nn as nn
class TextBERTClassifier(nn.Module):
    def __init__(self,d=768,dropout=.1): super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(dropout),nn.Linear(256,1))
    def forward(self,t,a=None,v=None,mask=None):
        idx=mask.sum(1).clamp(min=1)-1; b=torch.arange(t.size(0),device=t.device); return self.net(t[b,idx]).squeeze(-1)
class EarlyFusionTransformer(nn.Module):
    def __init__(self,td=768,ad=80,vd=512,d=256,heads=4,layers=2,ff=512,dropout=.1):
        super().__init__(); self.p=nn.Linear(td+ad+vd,d); e=nn.TransformerEncoderLayer(d,heads,ff,dropout,batch_first=True); self.tr=nn.TransformerEncoder(e,layers); self.c=nn.Linear(d,1)
    def forward(self,t,a,v,mask=None):
        z=self.tr(self.p(torch.cat([t,a,v],-1)),src_key_padding_mask=~mask); idx=mask.sum(1)-1;b=torch.arange(t.size(0),device=t.device);return self.c(z[b,idx]).squeeze(-1)
class MISA(nn.Module):
    """Author reimplementation inspired by modality-specific/shared representation separation; not original authors' source code."""
    def __init__(self,td=768,ad=80,vd=512,d=256,dropout=.1):
        super().__init__(); self.pt=nn.Linear(td,d);self.pa=nn.Linear(ad,d);self.pv=nn.Linear(vd,d);self.shared=nn.Linear(d,d);self.spec=nn.ModuleList([nn.Linear(d,d) for _ in range(3)]);self.c=nn.Sequential(nn.Linear(d*6,512),nn.ReLU(),nn.Dropout(dropout),nn.Linear(512,1))
    def forward(self,t,a,v,mask=None):
        idx=mask.sum(1)-1;b=torch.arange(t.size(0),device=t.device); xs=[self.pt(t[b,idx]),self.pa(a[b,idx]),self.pv(v[b,idx])]; z=torch.cat([self.shared(x) for x in xs]+[self.spec[i](x) for i,x in enumerate(xs)],-1);return self.c(z).squeeze(-1)
class MMBT(nn.Module):
    """Author reimplementation of a compact multimodal bi-transformer-style comparator; not original authors' source code."""
    def __init__(self,td=768,ad=80,vd=512,d=256,heads=4,layers=2,ff=512,dropout=.1):
        super().__init__();self.pt=nn.Linear(td,d);self.pa=nn.Linear(ad,d);self.pv=nn.Linear(vd,d);e=nn.TransformerEncoderLayer(d,heads,ff,dropout,batch_first=True);self.tr=nn.TransformerEncoder(e,layers);self.c=nn.Linear(d,1)
    def forward(self,t,a,v,mask=None):
        idx=mask.sum(1)-1;b=torch.arange(t.size(0),device=t.device);tok=torch.stack([self.pt(t[b,idx]),self.pa(a[b,idx]),self.pv(v[b,idx])],1);return self.c(self.tr(tok).mean(1)).squeeze(-1)
