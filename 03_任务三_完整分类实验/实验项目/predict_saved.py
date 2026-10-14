"""Recreate task 4 from SAVED PCA and model, without training."""
import pickle
import numpy as np
import scipy.io as sio
import torch
from model import HybridSN
from run_experiments import ROOT, full_map, savejson

if __name__=='__main__':
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x=sio.loadmat(ROOT/'data/PaviaU.mat')['paviaU']
    gt=sio.loadmat(ROOT/'data/PaviaU_gt.mat')['paviaU_gt']
    with open(ROOT/'results/pca.pkl','rb') as f:pca=pickle.load(f)
    xp=pca.transform(x.reshape(-1,103).astype(float)).reshape(610,340,15).astype(np.float32)
    padded=np.pad(xp,((12,12),(12,12),(0,0)))
    net=HybridSN().to(device)
    net.load_state_dict(torch.load(ROOT/'results/runs/lr_0.001/best_model.pth',map_location=device,weights_only=True))
    test=np.load(ROOT/'results/split.npz')['test_coords']
    previous=np.load(ROOT/'results/task4/prediction_full.npy').copy()
    full_map(net,x,gt,padded,test,device)
    current=np.load(ROOT/'results/task4/prediction_full.npy')
    assert np.array_equal(previous,current)
    savejson(ROOT/'results/reload_check.json',dict(passed=True,all_pixels_equal=True,pixels=610*340,meaning='Separate process reloaded saved PCA and saved model and reproduced all map pixels.'))
    print('RELOAD CHECK PASSED',flush=True)
