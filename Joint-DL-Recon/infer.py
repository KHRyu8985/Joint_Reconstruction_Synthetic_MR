
import numpy as np
import h5py
import sys
import os
import data_loader as D
import argparse
from JVS_net_v2 import *
# File_list: list of files for inference (subjectwise)
file_list = ['Testing_Data/' + filenames for filenames in os.listdir('Testing_Data/') if filenames.startswith('J')]
print(file_list)

def vs_net_infer(filepath, net, weight, input_type='loraks', device_num=0):
    """
    vs_net_infer(filepath, net, weight, input_type='loraks', device_num=0):
    ====== INPUTs =====================
    filepath : path to testing h5 file
    net : network (before putting in GPU)
    weight: path to trained network weight
    input_type: For JPI-JVS-Net input is 'loraks', for JVS-Net, input is 'zpad'
    device_num: what GPU are you using? 0,1, etc...  
    ====== OUTPUTs ====================
    X_DL : DL output
    X : JPI or ZPAD res (input images)
    Y : True res (true images)
    """

    device = 'cuda:' + str(device_num)
    if input_type is 'loraks':
        print("input is from LORAKS")
        testset = D.MAGIC_Dataset_LORAKS(train_path,augmentation=False,verbosity=False)
    elif input_type is 'zpad':
        print("input is from Zero-Padding")
        testset = D.MAGIC_Dataset_zpad(train_path,augmentation=False,verbosity=False)
        
    net_gpu = net.to(device)
    if chpoint is not None:
        print('Loading network from:',chpoint)
        net_gpu.load_state_dict(torch.load(chpoint))
        net_gpu.eval()
    else:
        print('weight not inputted')
        return

    kbar = pkbar.Kbr(target = len(testloader), width=5)
    ########## Running Testing ####################

    for im0, true, tSens, tX_kJVC, tmask in enumerate(testloader):

        im0, true, tX_kJVC, tSens, tmask = im0.to(device,non_blocking=True), true.to(device,non_blocking=True), tX_kJVC.to(device,non_blocking=True),\
                                            tSens.to(device,non_blocking=True), tmask.to(device,non_blocking=True)


        with torch.no_grad():
            out = net_gpu(im0,tX_kJVC,tmask,tSens)
            out = out.detach().cpu().numpy()
        

if __name__ == "__main__":
    vs_net_train(args)
