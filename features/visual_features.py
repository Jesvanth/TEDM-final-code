import cv2, torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18FeatureExtractor:
    def __init__(self, device=None, frames=8):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); self.frames=frames
        w=ResNet18_Weights.DEFAULT; m=resnet18(weights=w); m.fc=torch.nn.Identity()
        self.model=m.to(self.device).eval(); self.pre=w.transforms()
    def _frames(self,path):
        cap=cv2.VideoCapture(str(path)); n=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),1)
        ids=set(int(x) for x in torch.linspace(0,n-1,self.frames).tolist()); out=[]; i=0
        while True:
            ok,f=cap.read()
            if not ok: break
            if i in ids: out.append(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)))
            i+=1
        cap.release(); return out
    @torch.inference_mode()
    def __call__(self,path):
        fs=self._frames(path)
        if not fs: return torch.zeros(512)
        x=torch.stack([self.pre(im) for im in fs]).to(self.device)
        return self.model(x).mean(0).cpu()
