import argparse,subprocess,sys,yaml
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--split-prefix',required=True);p.add_argument('--config',default='configs/tedm.yaml');p.add_argument('--out',default='outputs/ablations');a=p.parse_args();cfg=yaml.safe_load(open(a.config))
for ab in ['full','no_drift','no_audio','no_video','no_attention']:
 for s in cfg['training']['seeds']:
  subprocess.run([sys.executable,'-m','training.train','--manifest',a.manifest,'--train-ids',a.split_prefix+'_train.txt','--val-ids',a.split_prefix+'_val.txt','--test-ids',a.split_prefix+'_test.txt','--model','tedm','--ablation',ab,'--seed',str(s),'--config',a.config,'--out',f'{a.out}/{ab}/seed_{s}'],check=True)
