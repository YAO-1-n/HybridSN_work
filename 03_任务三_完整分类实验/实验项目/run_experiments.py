"""Run locally: python run_experiments.py --epochs 100
No downloads or installations. All paths are relative to this project.
"""
from pathlib import Path
import argparse, hashlib, json, pickle, random, time, platform, sys
import numpy as np
import scipy, scipy.io as sio
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from model import HybridSN

ROOT=Path(__file__).resolve().parent
NAMES=['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
COLORS=['#ffffff','#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']
CMAP=ListedColormap(COLORS); NORM=BoundaryNorm(np.arange(-.5,10.5,1),10)

def savejson(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def seed_all(seed=666):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

class Patches(Dataset):
    def __init__(self,padded,coords,gt): self.x=padded; self.coords=coords; self.gt=gt
    def __len__(self): return len(self.coords)
    def __getitem__(self,i):
        r,c=self.coords[i]
        x=np.ascontiguousarray(self.x[r:r+25,c:c+25].transpose(2,0,1))
        return torch.from_numpy(x), int(self.gt[r,c])-1

def loader(padded,coords,gt,shuffle=False):
    return DataLoader(Patches(padded,coords,gt),batch_size=256,shuffle=shuffle,num_workers=0,pin_memory=torch.cuda.is_available())

@torch.no_grad()
def predict(net,ld,device):
    net.eval(); yp=[]; yt=[]
    for x,y in ld:
        yp.append(net(x.to(device,non_blocking=True)).argmax(1).cpu().numpy()); yt.append(y.numpy())
    return np.concatenate(yp),np.concatenate(yt)

def metrics(y,p):
    return dict(OA=float(accuracy_score(y,p)),AA=float(recall_score(y,p,average='macro',labels=np.arange(9),zero_division=0)),Kappa=float(cohen_kappa_score(y,p)))

def setup():
    t=time.time(); out=ROOT/'results'; out.mkdir(exist_ok=True)
    x=sio.loadmat(ROOT/'data/PaviaU.mat')['paviaU']; gt=sio.loadmat(ROOT/'data/PaviaU_gt.mat')['paviaU_gt']
    coords=np.argwhere(gt>0); y=gt[coords[:,0],coords[:,1]]
    train,rest=train_test_split(coords,train_size=.3,stratify=y,random_state=666)
    val,test=train_test_split(rest,train_size=1/7,stratify=gt[rest[:,0],rest[:,1]],random_state=666)
    sets=[set(map(tuple,a)) for a in [train,val,test]]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    np.savez_compressed(out/'split.npz',train_coords=train,val_coords=val,test_coords=test)
    # Fit only on training-center spectra. No test labels are used here.
    pca=PCA(n_components=15,whiten=True,svd_solver='full')
    pca.fit(x[train[:,0],train[:,1]].astype(np.float64))
    xp=pca.transform(x.reshape(-1,x.shape[2]).astype(np.float64)).reshape(*gt.shape,15).astype(np.float32)
    with open(out/'pca.pkl','wb') as f: pickle.dump(pca,f)
    np.savez_compressed(out/'pca_parameters.npz',mean=pca.mean_,components=pca.components_,explained_variance=pca.explained_variance_)
    padded=np.pad(xp,((12,12),(12,12),(0,0)))
    split_hash=hashlib.sha256(b''.join(a.tobytes() for a in [train,val,test])).hexdigest()
    env=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,platform=platform.platform())
    savejson(out/'environment.json',env)
    savejson(out/'protocol.json',dict(dataset='Pavia University',shape=list(x.shape),n_labels=int((gt>0).sum()),class_counts={str(i):int((gt==i).sum()) for i in range(10)},train=len(train),val=len(val),test=len(test),seed=666,split_sha256=split_hash,PCA_fit='training centers only',PCA_components=15,whiten=True,svd_solver='full',patch=25,padding='constant zero',validation_every=5,selection='highest validation OA; earliest epoch on ties',test_evaluation='once after selection for each prespecified learning rate',spatial_limitation='random centers are disjoint, but adjacent patches overlap; not a held-out spatial-region benchmark',pca_variance=float(pca.explained_variance_ratio_.sum()),setup_seconds=time.time()-t))
    # Data figures: raw spectral values, not reflectance with invented units.
    fig,ax=plt.subplots(figsize=(9,4))
    for cid in [2,4,6]:
        spec=x[gt==cid].astype(float); mean=spec.mean(0); np.savetxt(out/f'spectrum_class_{cid}.csv',np.column_stack([np.arange(1,104),mean]),delimiter=',',header='band_number,mean_raw_value',comments='')
        ax.plot(np.arange(1,104),mean,label=NAMES[cid-1])
    ax.set(xlabel='Band number (1-103)',ylabel='Mean raw spectral value',title='Pavia University: class mean spectra'); ax.legend(); fig.tight_layout();fig.savefig(out/'spectral_curves.png',dpi=150);plt.close(fig)
    print('SETUP',len(train),len(val),len(test),split_hash,flush=True)
    return x,gt,padded,train,val,test,split_hash

def train_run(padded,gt,train,val,test,lr,epochs,device,split_hash,prefix='runs'):
    rd=ROOT/'results'/prefix/f'lr_{lr:g}';rd.mkdir(parents=True,exist_ok=True)
    seed_all(); net=HybridSN().to(device)
    init_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-6); lossfn=torch.nn.CrossEntropyLoss()
    trld=loader(padded,train,gt,True); valld=loader(padded,val,gt); history=[]; best=-1; best_epoch=0;t0=time.time()
    for ep in range(1,epochs+1):
        net.train();loss_sum=0;correct=0;total=0
        for a,b in trld:
            a,b=a.to(device,non_blocking=True),b.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); logits=net(a);loss=lossfn(logits,b)
            assert torch.isfinite(loss), 'nonfinite loss'
            loss.backward();opt.step()
            loss_sum+=loss.item()*len(b);correct+=(logits.argmax(1)==b).sum().item();total+=len(b)
        va=None
        if ep%5==0 or ep==epochs:
            p,y=predict(net,valld,device);va=float(accuracy_score(y,p))
            if va>best:
                best=va;best_epoch=ep;torch.save(net.state_dict(),rd/'best_model.pth')
        history.append(dict(epoch=ep,loss=loss_sum/total,train_accuracy=correct/total,val_accuracy=va))
        savejson(rd/'history.json',history)
        if ep%5==0 or ep==1: print(f'LR={lr:g} epoch={ep}/{epochs} loss={loss_sum/total:.5f} val={va} elapsed={(time.time()-t0)/60:.2f}min',flush=True)
    net.load_state_dict(torch.load(rd/'best_model.pth',map_location=device,weights_only=True))
    p,y=predict(net,loader(padded,test,gt),device)
    np.savez_compressed(rd/'test_predictions.npz',coords=test,pred=p,true=y)
    cm=confusion_matrix(y,p,labels=np.arange(9));np.savetxt(rd/'confusion_counts.csv',cm,delimiter=',',fmt='%d')
    rep=classification_report(y,p,labels=np.arange(9),target_names=NAMES,digits=6,zero_division=0)
    (rd/'classification_report.txt').write_text(rep,encoding='utf-8')
    savejson(rd/'per_class.json',classification_report(y,p,labels=np.arange(9),target_names=NAMES,output_dict=True,zero_division=0))
    result=dict(lr=lr,epochs=epochs,batch=256,optimizer='Adam',weight_decay=1e-6,seed=666,best_epoch=best_epoch,best_validation_OA=best,train_minutes=(time.time()-t0)/60,split_sha256=split_hash,initial_weights_sha256=init_hash,params=sum(v.numel() for v in net.parameters()),**metrics(y,p))
    savejson(rd/'results.json',result)
    fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].plot([h['epoch'] for h in history],[h['loss'] for h in history]);axs[0].set(xlabel='Epoch',ylabel='Training loss')
    axs[1].plot([h['epoch'] for h in history],[h['train_accuracy'] for h in history],label='Training')
    hv=[h for h in history if h['val_accuracy'] is not None];axs[1].plot([h['epoch'] for h in hv],[h['val_accuracy'] for h in hv],label='Validation');axs[1].set(xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.02));axs[1].legend();fig.suptitle(f'Pavia University | lr={lr:g}');fig.tight_layout();fig.savefig(rd/'training_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(cm,cmap='Blues')
    for i in range(9):
        for j in range(9):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=range(9),yticks=range(9),xticklabels=NAMES,yticklabels=NAMES,xlabel='Predicted class',ylabel='True class',title=f'Test confusion counts | lr={lr:g}');plt.setp(ax.get_xticklabels(),rotation=60,ha='right');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(rd/'confusion_matrix.png',dpi=150);plt.close(fig)
    print('FINISHED',result,flush=True)
    return net,result

def full_map(net,x,gt,padded,test,device):
    out=ROOT/'results/task4';out.mkdir(exist_ok=True)
    coords=np.indices(gt.shape).reshape(2,-1).T
    p,_=predict(net,loader(padded,coords,gt),device);pred=(p+1).reshape(gt.shape).astype(np.uint8)
    np.save(out/'prediction_full.npy',pred);np.save(out/'prediction_labeled.npy',np.where(gt>0,pred,0).astype(np.uint8))
    saved=np.load(ROOT/'results/runs/lr_0.001/test_predictions.npz');assert np.array_equal(pred[test[:,0],test[:,1]]-1,saved['pred'])
    rgb=x[:,:,[50,30,10]].astype(float)
    for k in range(3):
        lo,hi=np.percentile(rgb[:,:,k],[2,98]);rgb[:,:,k]=np.clip((rgb[:,:,k]-lo)/(hi-lo),0,1)
    views=[rgb,gt,pred,np.where(gt>0,pred,0)];titles=['False color: bands 51/31/11','Ground truth (0 = unlabeled)','Full prediction (all pixels)','Prediction in labeled areas']
    for name,v,title in zip(['false_color','ground_truth','prediction_full','prediction_labeled'],views,titles):
        fig,ax=plt.subplots(figsize=(5,8));ax.imshow(v,**({} if v.ndim==3 else dict(cmap=CMAP,norm=NORM,interpolation='nearest')));ax.set_title(title);ax.axis('off');fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(16,9))
    for ax,v,title in zip(axs,views,titles):ax.imshow(v,**({} if v.ndim==3 else dict(cmap=CMAP,norm=NORM,interpolation='nearest')));ax.set_title(title,fontsize=10);ax.axis('off')
    fig.legend(handles=[Patch(color=COLORS[i],label=f'{i}: '+('Unlabeled' if i==0 else NAMES[i-1])) for i in range(10)],loc='lower center',ncol=5);fig.subplots_adjust(bottom=.15);fig.savefig(out/'classification_panel.png',dpi=170,bbox_inches='tight');plt.close(fig)
    yp=pred[test[:,0],test[:,1]];yt=gt[test[:,0],test[:,1]];bad=yp!=yt
    errors=np.column_stack([test[bad],yt[bad],yp[bad]]);np.savetxt(out/'test_errors.csv',errors,delimiter=',',fmt='%d',header='row,col,true_id,pred_id',comments='')
    # Boundary: at least one 4-connected labeled neighbor has a different class.
    boundary=np.zeros(gt.shape,bool)
    d=(gt[:-1]>0)&(gt[1:]>0)&(gt[:-1]!=gt[1:]);boundary[:-1]|=d;boundary[1:]|=d
    d=(gt[:,:-1]>0)&(gt[:,1:]>0)&(gt[:,:-1]!=gt[:,1:]);boundary[:,:-1]|=d;boundary[:,1:]|=d
    tb=boundary[test[:,0],test[:,1]]
    stats=dict(n_test=len(test),n_errors=int(bad.sum()),boundary_definition='4-connected, both pixels labeled, labels differ; excludes unlabeled edges',boundary_n=int(tb.sum()),boundary_errors=int((bad&tb).sum()),interior_n=int((~tb).sum()),interior_errors=int((bad&~tb).sum()),boundary_error_rate=float(bad[tb].mean()) if tb.any() else None,interior_error_rate=float(bad[~tb].mean()),map_shape=list(pred.shape),map_range=[int(pred.min()),int(pred.max())],map_test_consistency=True)
    from scipy.ndimage import label
    structure=np.array([[0,1,0],[1,1,1],[0,1,0]])
    sizes=np.zeros(gt.shape,np.int32)
    for cid in range(1,10):
        components,_=label(gt==cid,structure=structure);count=np.bincount(components.ravel());sizes[gt==cid]=count[components[gt==cid]]
    small=sizes[test[:,0],test[:,1]]<=25
    stats.update(small_region_definition='4-connected same-class labeled component <=25 pixels; GT annotation fragmentation is not physical field size',small_region_n=int(small.sum()),small_region_errors=int((bad&small).sum()),small_region_error_rate=float(bad[small].mean()) if small.any() else None,large_region_n=int((~small).sum()),large_region_errors=int((bad&~small).sum()),large_region_error_rate=float(bad[~small].mean()) if (~small).any() else None)
    savejson(out/'error_analysis.json',stats)
    selected=errors[:6];savejson(out/'crop_coordinates.json',[dict(row=int(r),col=int(c),true_id=int(t),pred_id=int(p)) for r,c,t,p in selected])
    if len(selected):
        fig,axs=plt.subplots(len(selected),3,figsize=(10,3*len(selected)),squeeze=False)
        for i,(r,c,t,pp) in enumerate(selected):
            r0,r1=max(0,r-25),min(gt.shape[0],r+26);c0,c1=max(0,c-25),min(gt.shape[1],c+26)
            for j,v in enumerate([rgb,gt,pred]):
                a=v[r0:r1,c0:c1];axs[i,j].imshow(a,**({} if a.ndim==3 else dict(cmap=CMAP,norm=NORM,interpolation='nearest')));axs[i,j].plot(c-c0,r-r0,'k+',markersize=14);axs[i,j].axis('off');axs[i,j].set_title(f'({r},{c}) true={t} pred={pp} | '+['False color','GT','Prediction'][j],fontsize=9)
        fig.tight_layout();fig.savefig(out/'error_crops.png',dpi=130);plt.close(fig)
    print('MAP COMPLETE',stats,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    seed_all();torch.set_num_threads(4)
    x,gt,padded,train,val,test,h=setup();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.smoke:
        _,res=train_run(padded,gt,train,val,test,.001,2,device,h,'smoke');savejson(ROOT/'results/smoke_check.json',dict(passed=True,meaning='2 epochs only; pipeline verification, not final experiment',result=res));return
    results=[]
    for lr in [.0001,.001,.01]:
        net,res=train_run(padded,gt,train,val,test,lr,a.epochs,device,h);results.append(res)
        if lr==.001:full_map(net,x,gt,padded,test,device)
        del net;torch.cuda.empty_cache()
    assert len({r['split_sha256'] for r in results})==1
    assert len({r['initial_weights_sha256'] for r in results})==1
    savejson(ROOT/'results/comparison.json',results)
    import csv
    with open(ROOT/'results/comparison.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['Learning rate','OA (%)','AA (%)','Kappa (0-1)','Best epoch','Minutes'])
        for r in results:w.writerow([r['lr'],r['OA']*100,r['AA']*100,r['Kappa'],r['best_epoch'],r['train_minutes']])
    print('ALL THREE COMPLETE',flush=True)

if __name__=='__main__':main()

