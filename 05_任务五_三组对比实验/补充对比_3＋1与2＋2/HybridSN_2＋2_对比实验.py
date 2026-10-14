"""3+1 baseline and modified 2+2: single-file publication copy.
Merged from existing model/dataset/training code. No new experimental results.
Data are external; new executions save to a separate output directory.
Original checkpoint/source hashes belong to the historical experiment.
"""
"""HybridSN and a controlled 3D-to-2D layer replacement.
Architecture adapted from user's local implementation of Roy et al.
DOI: 10.1109/LGRS.2019.2918719. PyTorch order: depth, height, width.
"""
import torch
from torch import nn

ARCHITECTURES = ('3d3_2d1', '3d2_2d2')

class HybridSN(nn.Module):
    def __init__(self, components, classes, architecture='3d3_2d1'):
        super().__init__()
        if architecture not in ARCHITECTURES or components < 13:
            raise ValueError('Invalid architecture or PCA depth')
        self.architecture = architecture
        self.conv1 = nn.Sequential(nn.Conv3d(1, 8, (7, 3, 3)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(8, 16, (5, 3, 3)), nn.ReLU())
        if architecture == '3d3_2d1':
            self.conv3 = nn.Sequential(nn.Conv3d(16, 32, (3, 3, 3)), nn.ReLU())
            self.conv4 = nn.Sequential(nn.Conv2d(32*(components-12), 64, 3), nn.ReLU())
        else:
            self.conv3 = nn.Sequential(nn.Conv2d(16*(components-10), 32, 3), nn.ReLU())
            self.conv4 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(64*17*17, 256), nn.ReLU(), nn.Dropout(.4))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.4))
        self.fc3 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv2(self.conv1(x.unsqueeze(1)))
        if self.architecture == '3d3_2d1':
            x = self.conv3(x)
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        else:
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
            x = self.conv3(x)
        x = self.conv4(x).flatten(1)
        return self.fc3(self.fc2(self.fc1(x)))


from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent/'data'
DATASETS = {
 'IP': dict(name='Indian Pines', chinese='印第安松树农田场景', data='Indian_pines_corrected.mat', key='indian_pines_corrected', gt='Indian_pines_gt.mat', gt_key='indian_pines_gt', components=30,
   names=['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees','Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill','Soybean-clean','Wheat','Woods','Buildings-Grass-Trees-Drives','Stone-Steel-Towers'],
   zh=['苜蓿','免耕玉米','少耕玉米','玉米','牧草','草地与树木','已割牧草','成行干草','燕麦','免耕大豆','少耕大豆','清耕大豆','小麦','树林','建筑物／草／树／车道混合区域','石材／钢塔区域']),
 'UP': dict(name='Pavia University', chinese='帕维亚大学城市场景', data='PaviaU.mat', key='paviaU', gt='PaviaU_gt.mat', gt_key='paviaU_gt', components=15,
   names=['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows'],
   zh=['沥青','草地','砾石','树木','涂漆金属板','裸土','柏油','自锁砖','阴影']),
 'SA': dict(name='Salinas Scene', chinese='萨利纳斯农业场景', data='Salinas_corrected.mat', key='salinas_corrected', gt='Salinas_gt.mat', gt_key='salinas_gt', components=15,
   names=['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth','Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds','Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis'],
   zh=['西兰花与杂草1','西兰花与杂草2','休耕地','粗犁休耕地','平整休耕地','作物残茬','芹菜','未整枝葡萄区域','葡萄园土壤区域','衰老玉米与绿色杂草','4周龄罗马生菜','5周龄罗马生菜','6周龄罗马生菜','7周龄罗马生菜','未整枝葡萄园','垂直棚架葡萄园']),
}


"""12 prespecified experiments; no downloads or automatic setting reductions."""
from pathlib import Path
import argparse, csv, hashlib, json, os, pickle, platform, random, shutil, sys, time, traceback
from datetime import datetime, timezone, timedelta
import numpy as np
import scipy, scipy.io as sio
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, confusion_matrix, classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
TZ = timezone(timedelta(hours=8))
BATCH, SEED, LR, WD = 256, 666, .001, 1e-6
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})

def stamp(): return datetime.now(TZ).isoformat(timespec='seconds')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def jsonsave(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)
def checkpoint(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)
def sync(): torch.cuda.synchronize()
def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def score(y,p,n):
    return {'OA':float(accuracy_score(y,p)),'AA':float(recall_score(y,p,average='macro',labels=np.arange(n),zero_division=0)),'Kappa':float(cohen_kappa_score(y,p))}

class Patches(Dataset):
    def __init__(self,padded,coords,gt):self.x=padded;self.coords=coords;self.gt=gt
    def __len__(self):return len(self.coords)
    def __getitem__(self,i):
        r,c=self.coords[i]
        x=np.ascontiguousarray(self.x[r:r+25,c:c+25].transpose(2,0,1))
        return torch.from_numpy(x),int(self.gt[r,c])-1
def loader(padded,coords,gt,shuffle=False,generator=None):
    return DataLoader(Patches(padded,coords,gt),batch_size=BATCH,shuffle=shuffle,num_workers=0,pin_memory=True,generator=generator)
@torch.no_grad()
def predict(net,ld):
    net.eval();pred=[];true=[]
    for x,y in ld:
        pred.append(net(x.cuda(non_blocking=True)).argmax(1).cpu().numpy());true.append(y.numpy())
    return np.concatenate(pred),np.concatenate(true)

def prepare(code,ratio):
    cfg=DATASETS[code];directory=RESULTS/'prepared'/code/f'train_{int(ratio*100)}';directory.mkdir(parents=True,exist_ok=True)
    x=sio.loadmat(DATA_ROOT/cfg['data'])[cfg['key']];gt=sio.loadmat(DATA_ROOT/cfg['gt'])[cfg['gt_key']]
    n=len(cfg['names']);assert x.shape[:2]==gt.shape and np.isfinite(x).all()
    assert np.array_equal(np.unique(gt),np.arange(n+1))
    coords=np.argwhere(gt>0);y=gt[coords[:,0],coords[:,1]]
    manifest={k:dict(path=str(DATA_ROOT/cfg[k]),sha256=sha(DATA_ROOT/cfg[k])) for k in ['data','gt']}
    if (directory/'protocol.json').exists():
        protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['sources']==manifest,'Source data changed'
        assert protocol['seed']==SEED and protocol['train_ratio']==ratio
        saved=np.load(directory/'split.npz');train,val,test=[saved[k] for k in ['train','validation','test']]
        with open(directory/'pca.pkl','rb') as f:pca=pickle.load(f)
        assert sha(directory/'pca.pkl')==protocol['pca_sha256']
    else:
        train,rest=train_test_split(coords,train_size=ratio,stratify=y,random_state=SEED)
        val,test=train_test_split(rest,train_size=.1/(1-ratio),stratify=gt[rest[:,0],rest[:,1]],random_state=SEED)
        pca=PCA(n_components=cfg['components'],whiten=True,svd_solver='full')
        pca.fit(x[train[:,0],train[:,1]].astype(np.float64))
        with open(directory/'pca.pkl','wb') as f:pickle.dump(pca,f)
        np.savez_compressed(directory/'pca_parameters.npz',mean=pca.mean_,components=pca.components_,explained_variance=pca.explained_variance_)
        np.savez_compressed(directory/'split.npz',train=train,validation=val,test=test)
        protocol=dict(dataset=code,name=cfg['name'],shape=list(x.shape),classes=n,labeled=len(coords),train_ratio=ratio,validation_ratio=.1,seed=SEED,counts={k:len(a) for k,a in [('train',train),('validation',val),('test',test)]},per_class_counts={str(i):{k:int((gt[a[:,0],a[:,1]]==i).sum()) for k,a in [('train',train),('validation',val),('test',test)]} for i in range(1,n+1)},class_names=cfg['names'],class_names_chinese=cfg['zh'],sources=manifest,PCA_components=cfg['components'],PCA_fit='training-center spectra only',PCA_whiten=True,PCA_solver='full',PCA_variance=float(pca.explained_variance_ratio_.sum()),pca_sha256=sha(directory/'pca.pkl'),split_sha256=sha(directory/'split.npz'),patch=25,padding='constant zero at scene edges',selection='validation OA every 5 epochs; earliest on ties',spatial_limitation='disjoint random centers, overlapping neighboring patches; not spatially separated evaluation')
        jsonsave(directory/'protocol.json',protocol)
    sets=[set(map(tuple,a)) for a in [train,val,test]]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert len(train)+len(val)+len(test)==len(coords)
    assert all(len(np.unique(gt[a[:,0],a[:,1]]))==n for a in [train,val,test])
    assert sha(directory/'split.npz')==protocol['split_sha256']
    xp=pca.transform(x.reshape(-1,x.shape[2]).astype(np.float64)).reshape(*gt.shape,cfg['components']).astype(np.float32)
    padded=np.pad(xp,((12,12),(12,12),(0,0)));assert np.isfinite(padded).all()
    return x,gt,padded,train,val,test,protocol

def warmup(net,components):
    net.eval()
    with torch.no_grad():
        dummy=torch.zeros(2,components,25,25,device='cuda')
        for _ in range(3): assert torch.isfinite(net(dummy)).all()
    sync();net.train()

def run(code,ratio,architecture,data,epochs=100,smoke=False):
    x,gt,padded,train,val,test,protocol=data;cfg=DATASETS[code];n=len(cfg['names'])
    rd=RESULTS/('smoke' if smoke else 'runs')/code/f'train_{int(ratio*100)}'/architecture;rd.mkdir(parents=True,exist_ok=True)
    config=dict(dataset=code,train_ratio=ratio,architecture=architecture,epochs=epochs,batch=BATCH,seed=SEED,lr=LR,weight_decay=WD,optimizer='Adam',dropout=.4,split_sha256=protocol['split_sha256'],pca_sha256=protocol['pca_sha256'],components=cfg['components'],classes=n,timing_source='local machine measured in this experiment; not paper timing')
    source_hashes={Path(__file__).name:sha(Path(__file__))}
    if (rd/'complete.json').exists():
        done=json.loads((rd/'results.json').read_text(encoding='utf-8'))
        assert done['config']==config and done['source_hashes']==source_hashes
        print('SKIP COMPLETE',code,ratio,architecture,flush=True);return done
    if not (rd/'config.json').exists():
        jsonsave(rd/'config.json',config);snapshot=rd/'source_used';snapshot.mkdir(exist_ok=True)
        for f in source_hashes:shutil.copy2(ROOT/f,snapshot/f)
    assert json.loads((rd/'config.json').read_text(encoding='utf-8'))==config
    assert all(sha(rd/'source_used'/f)==source_hashes[f] for f in source_hashes),'Changed source; refusing mixed experiment'
    seed_all();net=HybridSN(cfg['components'],n,architecture).cuda();opt=torch.optim.Adam(net.parameters(),lr=LR,weight_decay=WD);lossfn=nn.CrossEntropyLoss()
    gen=torch.Generator().manual_seed(SEED);trld=loader(padded,train,gt,True,gen);vald=loader(padded,val,gt)
    warmup(net,cfg['components'])
    start=1;history=[];best=-1.;best_epoch=0;training_seconds=validation_seconds=checkpoint_seconds=wall_seconds=0.;started=stamp()
    if (rd/'last_checkpoint.pth').exists():
        saved=torch.load(rd/'last_checkpoint.pth',map_location='cpu',weights_only=False)
        assert saved['config']==config and saved['source_hashes']==source_hashes
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        start=saved['epoch']+1;history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];started=saved['started']
        training_seconds=saved['training_seconds'];validation_seconds=saved['validation_seconds'];checkpoint_seconds=saved['checkpoint_seconds'];wall_seconds=saved['wall_seconds']
        random.setstate(saved['random']);np.random.set_state(saved['numpy']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);gen.set_state(saved['loader_rng'])
        print('RESUME',code,ratio,architecture,start,flush=True)
    print('START',code,ratio,architecture,'epochs',epochs,'time',stamp(),flush=True)
    for ep in range(start,epochs+1):
        epoch_wall=time.perf_counter();net.train();total=correct=0;loss_sum=0.
        sync();t=time.perf_counter()
        for a,b in trld:
            a,b=a.cuda(non_blocking=True),b.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True);logits=net(a);loss=lossfn(logits,b)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; no automatic setting reduction')
            loss.backward();opt.step();loss_sum+=loss.item()*len(b);correct+=(logits.argmax(1)==b).sum().item();total+=len(b)
        sync();train_time=time.perf_counter()-t;training_seconds+=train_time;va=None;val_time=0.
        if ep%5==0 or ep==epochs:
            sync();t=time.perf_counter();p,y=predict(net,vald);sync();val_time=time.perf_counter()-t;validation_seconds+=val_time;va=float(accuracy_score(y,p))
            if va>best:
                best=va;best_epoch=ep;t=time.perf_counter();checkpoint(rd/'best_model.pth',net.state_dict());checkpoint_seconds+=time.perf_counter()-t
        history.append(dict(epoch=ep,loss=loss_sum/total,train_accuracy=correct/total,validation_accuracy=va,training_seconds=train_time,validation_seconds=val_time))
        jsonsave(rd/'history.json',history)
        wall_seconds+=time.perf_counter()-epoch_wall
        saved=dict(config=config,source_hashes=source_hashes,epoch=ep,model=net.state_dict(),optimizer=opt.state_dict(),history=history,best=best,best_epoch=best_epoch,started=started,training_seconds=training_seconds,validation_seconds=validation_seconds,checkpoint_seconds=checkpoint_seconds,wall_seconds=wall_seconds,random=random.getstate(),numpy=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=gen.get_state())
        t=time.perf_counter();checkpoint(rd/'last_checkpoint.pth',saved);dt=time.perf_counter()-t;checkpoint_seconds+=dt;wall_seconds+=dt
        jsonsave(RESULTS/'progress.json',dict(phase='smoke' if smoke else 'training',dataset=code,ratio=ratio,architecture=architecture,epoch=ep,total=epochs,updated=stamp()))
        if ep==1 or ep%5==0: print(f'{code} train={ratio:g} {architecture} epoch={ep}/{epochs} loss={loss_sum/total:.6f} val={va} train_local={training_seconds:.1f}s',flush=True)
    assert len(history)==epochs and [h['epoch'] for h in history]==list(range(1,epochs+1))
    net.load_state_dict(torch.load(rd/'best_model.pth',map_location='cuda',weights_only=True))
    sync();t=time.perf_counter();p,y=predict(net,loader(padded,test,gt));sync();test_seconds=time.perf_counter()-t
    np.savez_compressed(rd/'test_predictions.npz',coords=test,pred=p,true=y)
    cm=confusion_matrix(y,p,labels=np.arange(n));np.savetxt(rd/'confusion_counts.csv',cm,delimiter=',',fmt='%d')
    rep=classification_report(y,p,labels=np.arange(n),target_names=cfg['names'],digits=6,zero_division=0)
    (rd/'classification_report.txt').write_text(rep,encoding='utf-8')
    jsonsave(rd/'per_class.json',classification_report(y,p,labels=np.arange(n),target_names=cfg['names'],output_dict=True,zero_division=0))
    reloadnet=HybridSN(cfg['components'],n,architecture).cuda();reloadnet.load_state_dict(torch.load(rd/'best_model.pth',map_location='cuda',weights_only=True))
    rp,ry=predict(reloadnet,loader(padded,test,gt));assert np.array_equal(rp,p) and np.array_equal(ry,y);del reloadnet
    full_seconds=None
    if not smoke:
        coords=np.indices(gt.shape).reshape(2,-1).T
        sync();t=time.perf_counter();fp,_=predict(net,loader(padded,coords,gt));sync();full_seconds=time.perf_counter()-t
        full=(fp+1).reshape(gt.shape).astype(np.uint8)
        assert np.array_equal(full[test[:,0],test[:,1]]-1,p)
        np.save(rd/'prediction_full.npy',full);np.save(rd/'prediction_labeled.npy',np.where(gt>0,full,0).astype(np.uint8))
        errors=np.column_stack([test[p!=y],y[p!=y]+1,p[p!=y]+1]);np.savetxt(rd/'test_errors.csv',errors,delimiter=',',fmt='%d',header='row,col,true_id,predicted_id',comments='')
        palette=['#ffffff']+[matplotlib.colors.to_hex(plt.get_cmap('tab20')(i)) for i in range(n)]
        fig,ax=plt.subplots(figsize=(6,7));ax.imshow(np.where(gt>0,full,0),cmap=ListedColormap(palette),norm=BoundaryNorm(np.arange(-.5,n+1.5),n+1),interpolation='nearest');ax.set_title(f'{code} | {architecture} | 训练{int(ratio*100)}%\n有标签区域预测；成绩仅在测试位置计算');ax.axis('off');fig.tight_layout();fig.savefig(rd/'prediction_labeled.png',dpi=150);plt.close(fig)
    result=dict(config=config,source_hashes=source_hashes,started=started,finished=stamp(),epochs_completed=len(history),best_epoch=best_epoch,best_validation_OA=best,params=sum(v.numel() for v in net.parameters()),training_seconds=training_seconds,validation_seconds=validation_seconds,checkpoint_io_seconds=checkpoint_seconds,training_wall_seconds=wall_seconds,test_inference_seconds=test_seconds,test_count=len(test),test_pixels_per_second=len(test)/test_seconds,full_map_inference_seconds=full_seconds,full_map_count=int(gt.size) if not smoke else None,smoke=smoke,reload_prediction_identical=True,map_test_prediction_identical=True if not smoke else None,**score(y,p,n))
    jsonsave(rd/'results.json',result);plots(rd,history,cm,cfg,architecture,ratio)
    jsonsave(rd/'complete.json',dict(finished=stamp(),results_sha256=sha(rd/'results.json')))
    print('COMPLETE',code,ratio,architecture,'OA',result['OA'],'AA',result['AA'],'Kappa',result['Kappa'],'training_local_seconds',training_seconds,flush=True)
    del net,opt;torch.cuda.empty_cache();return result

def plots(rd,history,cm,cfg,architecture,ratio):
    fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].plot([h['epoch'] for h in history],[h['loss'] for h in history]);axs[0].set(xlabel='训练轮次',ylabel='训练损失')
    axs[1].plot([h['epoch'] for h in history],[h['train_accuracy'] for h in history],label='训练成绩');v=[h for h in history if h['validation_accuracy'] is not None];axs[1].plot([h['epoch'] for h in v],[h['validation_accuracy'] for h in v],label='验证成绩');axs[1].set(xlabel='训练轮次',ylabel='答对比例',ylim=(0,1.02));axs[1].legend();fig.suptitle(f'{cfg["name"]} | 训练{int(ratio*100)}% | {architecture}');fig.tight_layout();fig.savefig(rd/'training_curves.png',dpi=160);plt.close(fig)
    n=len(cfg['names']);fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(cm,cmap='Blues')
    for i in range(n):
        for j in range(n):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=range(n),yticks=range(n),xticklabels=[f'{i+1} {s}' for i,s in enumerate(cfg['zh'])],yticklabels=[f'{i+1} {s}' for i,s in enumerate(cfg['zh'])],xlabel='模型回答',ylabel='参考答案',title='测试集混淆矩阵：每格是像素数量');plt.setp(ax.get_xticklabels(),rotation=65,ha='right');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(rd/'confusion_matrix.png',dpi=150);plt.close(fig)

def inventory():
    assert torch.cuda.is_available(),'CUDA unavailable; refusing to silently switch to CPU'
    env=dict(recorded=stamp(),python=sys.version,python_path=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),timing_source='all timing from this local machine, not the paper')
    jsonsave(RESULTS/'environment.json',env)
    sources={name:{kind:dict(path=str(DATA_ROOT/cfg[kind]),sha256=sha(DATA_ROOT/cfg[kind])) for kind in ['data','gt']} for name,cfg in DATASETS.items()}
    jsonsave(RESULTS/'data_sources.json',dict(sources=sources,route='reused public datasets already on user machine; no download',source_reference='https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes',original_local_acquisition='not independently verified',raw_data_modified=False))

def main():
    global RESULTS
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS));ap.add_argument('--train-ratios',nargs='+',type=float,default=[.1,.3]);ap.add_argument('--architectures',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES));ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--output-dir',type=Path,default=ROOT/'重新运行结果');args=ap.parse_args();RESULTS=args.output_dir.resolve()
    assert all(r in [.1,.3] for r in args.train_ratios);assert args.epochs==100 or args.smoke,'Formal protocol requires 100 epochs'
    torch.set_num_threads(4);RESULTS.mkdir(exist_ok=True);inventory()
    for code in args.datasets:
        for ratio in ([.1] if args.smoke else args.train_ratios):
            data=prepare(code,ratio)
            for architecture in args.architectures:
                run(code,ratio,architecture,data,2 if args.smoke else args.epochs,args.smoke)
            del data
    jsonsave(RESULTS/('smoke_complete.json' if args.smoke else 'training_complete.json'),dict(finished=stamp(),datasets=args.datasets,ratios=[.1] if args.smoke else args.train_ratios,architectures=args.architectures))
    if not args.smoke:
        rows=[]
        for code in args.datasets:
            for ratio in args.train_ratios:
                for arch in args.architectures:
                    saved=json.loads((RESULTS/'runs'/code/f'train_{int(ratio*100)}'/arch/'results.json').read_text(encoding='utf-8'))
                    rows.append(dict(dataset=code,train_ratio=ratio,architecture=arch,OA_percent=saved['OA']*100,AA_percent=saved['AA']*100,Kappa=saved['Kappa'],best_epoch=saved['best_epoch'],epochs=saved['epochs_completed']))
        with (RESULTS/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('ALL FINISHED',stamp(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        jsonsave(RESULTS/'failure.json',dict(time=stamp(),traceback=traceback.format_exc()));raise
