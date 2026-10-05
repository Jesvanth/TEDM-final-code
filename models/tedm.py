import torch
import torch.nn as nn

class TEDM(nn.Module):
    def __init__(self,text_dim=768,audio_dim=80,video_dim=512,latent=256,gru_hidden=256,gru_layers=1,nhead=4,nlayers=2,ff=512,dropout=.1,use_drift=True,use_audio=True,use_video=True,use_attention=True):
        super().__init__(); self.use_drift=use_drift; self.use_audio=use_audio; self.use_video=use_video; self.use_attention=use_attention
        self.tp=nn.Linear(text_dim,latent); self.ap=nn.Linear(audio_dim,latent); self.vp=nn.Linear(video_dim,latent)
        self.gru=nn.GRU(latent*3,gru_hidden,gru_layers,batch_first=True)
        layer=nn.TransformerEncoderLayer(gru_hidden,nhead,ff,dropout,batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,nlayers)
        self.classifier=nn.Sequential(nn.Linear(gru_hidden*(2 if use_drift else 1),128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,1))
    def forward(self,t,a,v,mask=None,return_aux=False):
        t=self.tp(t); a=self.ap(a) if self.use_audio else torch.zeros_like(t); v=self.vp(v) if self.use_video else torch.zeros_like(t)
        h,_=self.gru(torch.cat([t,a,v],-1)); drift=torch.zeros_like(h); drift[:,1:]=h[:,1:]-h[:,:-1]
        z=self.transformer(h,src_key_padding_mask=(~mask if mask is not None else None)) if self.use_attention else h
        idx=mask.sum(1).clamp(min=1)-1 if mask is not None else torch.full((z.size(0),),z.size(1)-1,device=z.device)
        b=torch.arange(z.size(0),device=z.device); base=z[b,idx]; d=drift[b,idx]
        logits=self.classifier(torch.cat([base,d],-1) if self.use_drift else base).squeeze(-1)
        if return_aux: return logits,{'drift':drift,'context':z}
        return logits
