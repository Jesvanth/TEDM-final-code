import torch
from models.tedm import TEDM
m=TEDM();t=torch.randn(2,4,768);a=torch.randn(2,4,80);v=torch.randn(2,4,512);mask=torch.ones(2,4,dtype=torch.bool);assert m(t,a,v,mask).shape==(2,)
print('shape test passed')
