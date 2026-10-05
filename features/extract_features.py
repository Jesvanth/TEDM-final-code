import argparse, os, torch, pandas as pd, yaml
from pathlib import Path
from features.text_features import BERTFeatureExtractor
from features.audio_features import extract_mfcc
from features.visual_features import ResNet18FeatureExtractor

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--config',default='configs/tedm.yaml'); p.add_argument('--out',default='data/features'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); m=cfg['model']; df=pd.read_csv(a.manifest); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    te=BERTFeatureExtractor(m['text_model']); ve=ResNet18FeatureExtractor()
    rows=[]
    for did,g in df.groupby('dialogue_id',sort=False):
        g=g.sort_values('turn_index'); T=[];A=[];V=[];Y=[]
        for _,r in g.iterrows():
            T.append(te(r['text'])); A.append(extract_mfcc(r['audio_path'],**{k:m['audio'][k] for k in ['sample_rate','n_mfcc','n_fft','win_length','hop_length','summary']})); V.append(ve(r['video_path'])); Y.append(int(r['label']))
        fp=out/f'{did}.pt'; torch.save({'text':torch.stack(T),'audio':torch.stack(A),'video':torch.stack(V),'labels':torch.tensor(Y),'dialogue_id':str(did)},fp)
        rows.append({'dialogue_id':did,'feature_path':str(fp),'label':Y[-1]})
    pd.DataFrame(rows).to_csv(out/'feature_manifest.csv',index=False)
if __name__=='__main__': main()
