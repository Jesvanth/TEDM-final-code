import argparse,subprocess,sys,yaml
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--split-prefix',required=True);p.add_argument('--config',default='configs/tedm.yaml');p.add_argument('--out',default='outputs');a=p.parse_args();cfg=yaml.safe_load(open(a.config))
for model in ['text_bert','misa','mmbt','early_fusion','tedm']:
 for s in cfg['training']['seeds']:
  cmd=[sys.executable,'-m','training.train','--manifest',a.manifest,'--train-ids',a.split_prefix+'_train.txt','--val-ids',a.split_prefix+'_val.txt','--test-ids',a.split_prefix+'_test.txt','--model',model,'--seed',str(s),'--config',a.config,'--out',f'{a.out}/{model}/seed_{s}'];subprocess.run(cmd,check=True)
