import argparse, json, random, numpy as np, torch, yaml
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import DialogueFeatures,collate
from models import TEDM,TextBERTClassifier,MISA,MMBT,EarlyFusionTransformer
from training.engine import run_epoch

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def build(name,cfg,ablation='full'):
    m=cfg['model']; kw=dict(text_dim=m['text_dim'],audio_dim=m['audio']['output_dim'],video_dim=m['video_dim'],latent=m['latent_dim'],gru_hidden=m['gru_hidden'],gru_layers=m['gru_layers'],nhead=m['attention_heads'],nlayers=m['transformer_layers'],ff=m['feedforward_dim'],dropout=m['dropout'])
    if name=='tedm': return TEDM(**kw,use_drift=ablation!='no_drift',use_audio=ablation!='no_audio',use_video=ablation!='no_video',use_attention=ablation!='no_attention')
    if name=='text_bert': return TextBERTClassifier(m['text_dim'],m['dropout'])
    if name=='misa': return MISA(m['text_dim'],m['audio']['output_dim'],m['video_dim'],m['latent_dim'],m['dropout'])
    if name=='mmbt': return MMBT(m['text_dim'],m['audio']['output_dim'],m['video_dim'],m['latent_dim'],m['attention_heads'],m['transformer_layers'],m['feedforward_dim'],m['dropout'])
    if name=='early_fusion': return EarlyFusionTransformer(m['text_dim'],m['audio']['output_dim'],m['video_dim'],m['latent_dim'],m['attention_heads'],m['transformer_layers'],m['feedforward_dim'],m['dropout'])
    raise ValueError(name)
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--train-ids',required=True);p.add_argument('--val-ids',required=True);p.add_argument('--test-ids',required=True);p.add_argument('--model',default='tedm',choices=['tedm','text_bert','misa','mmbt','early_fusion']);p.add_argument('--ablation',default='full');p.add_argument('--seed',type=int,default=11);p.add_argument('--config',default='configs/tedm.yaml');p.add_argument('--out',required=True);a=p.parse_args()
    cfg=yaml.safe_load(open(a.config));seed_all(a.seed);dev='cuda' if torch.cuda.is_available() else 'cpu';bs=cfg['training']['batch_size']
    dl=lambda f,sh=False:DataLoader(DialogueFeatures(a.manifest,f),batch_size=bs,shuffle=sh,collate_fn=collate)
    tr,va,te=dl(a.train_ids,True),dl(a.val_ids),dl(a.test_ids);model=build(a.model,cfg,a.ablation).to(dev);opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['training']['max_epochs'])
    best=1e99;wait=0;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ck=out/'best.pt'
    for ep in range(cfg['training']['max_epochs']):
        run_epoch(model,tr,dev,opt);vl,_,_,_=run_epoch(model,va,dev)
        if vl<best:best=vl;wait=0;torch.save(model.state_dict(),ck)
        else:wait+=1
        sch.step()
        if wait>=cfg['training']['early_stopping_patience']:break
    model.load_state_dict(torch.load(ck,map_location=dev));_,met,y,pred=run_epoch(model,te,dev);met.update({'seed':a.seed,'model':a.model,'ablation':a.ablation,'epochs':ep+1});json.dump({'metrics':met,'y_true':y,'y_prob':pred},open(out/'metrics.json','w'),indent=2);print(json.dumps(met,indent=2))
if __name__=='__main__':main()
