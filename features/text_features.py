import torch
from transformers import AutoTokenizer, AutoModel

class BERTFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModel.from_pretrained(model_name).to(self.device).eval()
    @torch.inference_mode()
    def __call__(self, text):
        x=self.tok(str(text), return_tensors='pt', truncation=True, max_length=128)
        x={k:v.to(self.device) for k,v in x.items()}
        return self.model(**x).last_hidden_state[:,0,:].squeeze(0).cpu()
