"""Normalize a dataset-specific CSV to TEDM's required utterance manifest.
Required columns after mapping: dialogue_id, turn_index, text, audio_path, video_path, label.
This script deliberately does not redistribute MUStARD/MUStARD++ media.
"""
import argparse,pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
for x in ['dialogue_id','turn_index','text','audio_path','video_path','label']: p.add_argument('--'+x)
a=p.parse_args(); df=pd.read_csv(a.input)
rename={getattr(a,x):x for x in ['dialogue_id','turn_index','text','audio_path','video_path','label'] if getattr(a,x)}; df=df.rename(columns=rename)
req=['dialogue_id','turn_index','text','audio_path','video_path','label']; miss=[x for x in req if x not in df]
if miss: raise SystemExit('Missing columns: '+','.join(miss))
df[req].to_csv(a.output,index=False)
