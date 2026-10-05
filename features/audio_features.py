import numpy as np, torch, librosa

def extract_mfcc(path, sample_rate=16000, n_mfcc=40, n_fft=512, win_length=400, hop_length=160, summary='mean_std'):
    y,_=librosa.load(path, sr=sample_rate, mono=True)
    if len(y)==0: y=np.zeros(win_length, dtype=np.float32)
    m=librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    if summary=='mean_std': feat=np.concatenate([m.mean(1),m.std(1)])
    elif summary=='mean': feat=m.mean(1)
    else: raise ValueError('summary must be mean_std or mean')
    return torch.tensor(feat,dtype=torch.float32)
