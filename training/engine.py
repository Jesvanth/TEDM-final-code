import torch
from training.metrics import metrics

def run_epoch(model,loader,device,opt=None):
    train=opt is not None; model.train(train); lossfn=torch.nn.BCEWithLogitsLoss(); total=0; ys=[];ps=[]
    for t,a,v,m,y in loader:
        t,a,v,m,y=[x.to(device) for x in (t,a,v,m,y)]; logits=model(t,a,v,m); loss=lossfn(logits,y)
        if train: opt.zero_grad();loss.backward();opt.step()
        total+=loss.item()*len(y);ys+=y.detach().cpu().tolist();ps+=torch.sigmoid(logits).detach().cpu().tolist()
    return total/max(len(loader.dataset),1),metrics(ys,ps),ys,ps
