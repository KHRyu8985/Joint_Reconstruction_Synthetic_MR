import matplotlib
matplotlib.use('Agg')
import argparse
import torch.nn as nn
import numpy as np
import math
import torch
import h5py
import os
import transforms as T
import data_loader as D
import torch.nn.functional as F

from livelossplot import PlotLosses

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

from JVS_net_v2 import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from network_utils import *
import pickle
import pdb
import pkbar

parser = argparse.ArgumentParser(description = 'Training MRM VS_net')
parser.add_argument('--train-h5','--t',type=str, help='Path for input h5 training')
parser.add_argument('--val-h5','--v',type=str, help='Path for input h5 validation')
parser.add_argument('--zpad','--z',type=bool, default=False, help='Zero_pad as input or not')
parser.add_argument('--cascade','--c',type=int, help='Cascade number for network')
parser.add_argument('--nb','--b',type=int, help='Number of Batchsize')
parser.add_argument('--epoch','--e',type=int, help='Epoch number for training')
parser.add_argument('--lr','--l',type=float, help='Learning rate for network')
parser.add_argument('--Result_name','--r',type=str, help='Saving to folder')
parser.add_argument('--device','--d',type=int, help='GPU Device Number')
parser.add_argument('--aug','--a',type=bool, default=True, help='Augmentation while Training')
parser.add_argument('--checkpoint','--ch', type=str, default = None, help='Path for checkpointing')
args = parser.parse_args()

def vs_net_train(args):
    train_path = args.train_h5
    val_path = args.val_h5
    NEPOCHS = args.epoch
    CASCADE = args.cascade
    LR = args.lr
    NBATCH = args.nb
    Res_name = args.Result_name
    device_num = args.device
    chpoint = args.checkpoint
    aug = args.aug
    zpad = args.zpad
        
    device = 'cuda:' + str(device_num)
    if zpad is False:
        print("input is from LORAKS")
        trainset = D.MAGIC_Dataset_LORAKS(train_path,augmentation=aug,verbosity=False)
        testset = D.MAGIC_Dataset_LORAKS(val_path,augmentation=False,verbosity=False)
    elif zpad is True:
        print("input is from Zero-Padding")
        trainset = D.MAGIC_Dataset_zpad(train_path,augmentation=aug,verbosity=False)
        testset = D.MAGIC_Dataset_zpad(val_path,augmentation=False,verbosity=False)
        
    trainloader = DataLoader(trainset, batch_size=NBATCH, shuffle=True, pin_memory=True,num_workers=0) 
    valloader = DataLoader(testset,batch_size=NBATCH, shuffle=False, pin_memory=True,num_workers=0)

    dataloaders = {
        'train': trainloader,
        'validation' : valloader
    }    
    
    net = network(alfa=None, beta=0.5, cascades=CASCADE)
    net = net.to(device)
    if chpoint is not None:
        print('Loading network from:',chpoint)
        net.load_state_dict(torch.load(chpoint))
    
    ########## Training ####################
    _im0, _true, _Sens,_X_kJVC, _mask = testset[13]
    
    _im0, _true, _Sens, _X_kJVC, _mask = _im0.unsqueeze(0).to(device), _true.unsqueeze(0).to(device), _Sens.unsqueeze(0).to(device),\
    _X_kJVC.unsqueeze(0).to(device), _mask.unsqueeze(0).to(device)

    
    criterion = torch.nn.L1Loss()
    
    liveloss = PlotLosses()
    optimizer = torch.optim.Adam(net.parameters(),lr=LR)
#    print('Now Training the Network')
#    pdb.set_trace()
    for epoch in range(NEPOCHS):
        print('Epoch', epoch+1)
        logs = {}
        for phase in {'train', 'validation'}:
            if phase == 'train':
                kbar = pkbar.Kbar(target = len(trainloader), width=2)
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_mse = 0.0
            
            iii = 0 
            for im0, true, tSens, tX_kJVC, tmask in dataloaders[phase]:

                im0, true, tX_kJVC, tSens, tmask = im0.to(device,non_blocking=True), true.to(device,non_blocking=True), tX_kJVC.to(device,non_blocking=True),\
                                                   tSens.to(device,non_blocking=True), tmask.to(device,non_blocking=True)

                if phase == 'train':
                    out = net(im0,tX_kJVC,tmask,tSens)
                    loss = criterion(out,true)                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss = running_loss + loss.item()  * im0.size(0)
                    prefix = ''
                    kbar.update(iii, values = [('L', 100 * running_loss/(iii+1))])
                    iii = iii + 1                                                
                else:
                    with torch.no_grad():
                        prefix = 'val_'
                        out = net(im0,tX_kJVC,tmask,tSens)                    
                        loss = criterion(out,true)
                        running_loss = running_loss + loss.item() * im0.size(0)
  #                  print('hello')
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset) 
                
                logs[prefix+'Loss'] = epoch_loss*100

        if epoch % 10 == 0:
            save_name = 'Result_' + Res_name +'/Val_Epoch_'+str(epoch)+'.jpg'
            show_output(net,_im0, _true, _X_kJVC, _Sens, _mask, save_name)
            file_name = 'models/'+ Res_name + '/Weights_Epoch_'+ str(epoch) 
            
            print(' SAVING WEIGHTS : ' + file_name)
            torch.save(net.state_dict(), file_name)

            f = open("models/" + Res_name + "/Losses_graph.obj","wb") # Saving Lossplot objects to pickle
            pickle.dump(liveloss,f)
            f.close()

        liveloss.update(logs)
        f = open("Loss_Logging.txt","a")
        kbar.add(1, values = [('Train',logs['Loss']),('Val', logs['val_Loss'])])
        f.write("Epoch{} : Training Loss : {:.5f} & Validation Loss: {:.5f}\n".format(epoch,logs['Loss'],logs['val_Loss']))
        f.close()
#        print('{} Training Loss: {:.5f} Validation Loss: {:.5f} '.format(epoch, logs['Loss'], #logs['val_Loss']))
        
if __name__ == "__main__":
    vs_net_train(args)
    


    
    
    
    


    
