"""Benchmarks the TEDM fusion/classification network on pre-extracted features.
BERT/ResNet extraction is intentionally reported separately because the manuscript pipeline uses pre-extracted embeddings.
"""
import argparse,time,torch,yaml,json
from models.tedm import TEDM
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/tedm.yaml');p.add_argument('--iters',type=int,default=200);a=p.parse_args();c=yaml.safe_load(open(a.config))['model'];dev='cuda' if torch.cuda.is_available() else 'cpu';m=TEDM(c['text_dim'],c['audio']['output_dim'],c['video_dim'],c['latent_dim'],c['gru_hidden'],c['gru_layers'],c['attention_heads'],c['transformer_layers'],c['feedforward_dim'],c['dropout']).to(dev).eval();n=sum(x.numel() for x in m.parameters());T=torch.randn(1,8,c['text_dim'],device=dev);A=torch.randn(1,8,c['audio']['output_dim'],device=dev);V=torch.randn(1,8,c['video_dim'],device=dev);M=torch.ones(1,8,dtype=torch.bool,device=dev)
with torch.inference_mode():
 for _ in range(20):m(T,A,V,M)
 if dev=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 t=time.perf_counter()
 for _ in range(a.iters):m(T,A,V,M)
 if dev=='cuda':torch.cuda.synchronize()
 dt=(time.perf_counter()-t)/a.iters; peak=torch.cuda.max_memory_allocated()/1024**2 if dev=='cuda' else None
print(json.dumps({'scope':'TEDM fusion network on pre-extracted features','parameters':n,'fp32_size_mb':n*4/1024**2,'latency_ms':dt*1000,'throughput_samples_s':1/dt,'peak_cuda_memory_mb':peak,'device':dev},indent=2))
