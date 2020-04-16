import glob, os, h5py
import numpy as np
from scipy.io import loadmat
import transforms as T
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import h5py
import tables

def combine_all_coils(image, sensitivity, coil_dim=0):
    """ Return Sensitivity combined images from all coils """
    combined = T.complex_multiply(sensitivity[...,0],
                                  -sensitivity[...,1],
                                  image[...,0],
                                  image[...,1])
    return combined.sum(dim=coil_dim)

def project_all_coils(x,sensitivity,coil_dim=1):
    """ Return combined image to coil images """
    x = T.complex_multiply(x[...,0].unsqueeze(coil_dim), x[...,1].unsqueeze(coil_dim), 
                       sensitivity[...,0], sensitivity[...,1])
    return x

class MAGIC_Dataset(Dataset):
    """ Dataloader that Load & Augment & change to Torch Tensors (For Training)
        Inputs : path to Dataset (.h5)
        Outputs : [X_JLORAKS, Y_JLORAKS, Sens, X_kJLORAKS, mask] """
    
    def __init__(self, h5_path,augmentation=False,verbosity=False):
        self.fname = h5_path
        self.tables = tables.open_file(self.fname)
        self.nslices = self.tables.root.X_JLORAKS.shape[0]
        self.tables.close()
        self.X_JLORAKS = None # reconstructed images from JLORAKS
        self.Y_JLORAKS = None # Fully sampled images
        self.Sens = None # Sensitivity maps 
        self.X_kJLORAKS = None # Acquired K-space 
        self.augmentation = augmentation # Whether augment the data 
        self.verbose = verbosity # Print out what is going on?
        
    def AugmentFlip(self,im, axis):
        im = im.swapaxes(0,axis)
        im = im[::-1]
        im = im.swapaxes(axis,0)
        return im.copy()
    
    def AugmentScale(self,im,scale_val):
        im = im * scale_val
        return im
    def __getitem__(self, ind):
        
        if(self.X_JLORAKS is None): # Open in thread
            self.tables = tables.open_file(self.fname, 'r')
            self.X_JLORAKS = self.tables.root.X_JLORAKS
            self.Y_JLORAKS = self.tables.root.Y_JLORAKS
            self.Sens = self.tables.root.Sens
            self.X_kJLORAKS = self.tables.root.X_kJLORAKS
            
#        t0 = time.time()
        '''
        X_JLORAKS = np.float32(np.array(self.X_JLORAKS[ind]))
        Y_JLORAKS = np.float32(np.array(self.Y_JLORAKS[ind]))
        Sens = np.float32(np.array(self.Sens[ind]))
        X_kJLORAKS = np.float32(np.array(self.X_kJLORAKS[ind]))
        '''
        X_JLORAKS = np.float32(self.X_JLORAKS[ind])
        Y_JLORAKS = np.float32(self.Y_JLORAKS[ind])
        Sens = np.float32(self.Sens[ind])
        X_kJLORAKS = np.float32(self.X_kJLORAKS[ind])
#        '''
        
#        print(time.time()-t0)
        
        """ Data Loading """
        mask = np.float32(np.abs(X_kJLORAKS)>0) 
        Sens = np.tile(Sens,[1,8,1,1,1])
        
        if self.verbose: 
            print('X_JLORAKS:',X_JLORAKS.shape, X_JLORAKS.dtype)
            print('Y_JLORAKS:', Y_JLORAKS.shape, Y_JLORAKS.dtype)
            print('Sens:', Sens.shape, Sens.dtype)
            print('X_kJLORAKS:', X_kJLORAKS.shape, X_kJLORAKS.dtype)
            print('mask:', mask.shape, mask.dtype)
        
        """ Augmentation (Random Flipping (None, Left-Right, Up-Down), Scaling (0.9 - 1.1) """
        if self.augmentation:
            """ Random Flipping """
            if np.random.random() < 0.5:
#                pdb.set_trace()
                X_JLORAKS = self.AugmentFlip(X_JLORAKS,1)
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS,1)
                Sens = self.AugmentFlip(Sens,2)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS,2)
                mask = self.AugmentFlip(mask,2)
                
            if np.random.random() < 0.5: 
                X_JLORAKS = self.AugmentFlip(X_JLORAKS,2)
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS,2)
                Sens = self.AugmentFlip(Sens,3)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS,3)
                mask = self.AugmentFlip(mask,3)
             
            scale_f = np.random.uniform(0.9,1.1)
            X_JLORAKS = self.AugmentScale(X_JLORAKS,scale_f)
            Y_JLORAKS = self.AugmentScale(Y_JLORAKS,scale_f)
            X_kJLORAKS = self.AugmentScale(X_kJLORAKS,scale_f)
                
        return torch.from_numpy(X_JLORAKS), torch.from_numpy(Y_JLORAKS),torch.from_numpy(Sens),torch.from_numpy(X_kJLORAKS), torch.from_numpy(mask)

 
    def __len__(self):
        return self.nslices
    
class MAGIC_Dataset_LORAKS(Dataset):
    """ Dataloader that Load & Augment & change to Torch Tensors (For Training)
        Inputs : path to Dataset (.h5)
        Outputs : [X_JLORAKS, Y_JLORAKS, Sens, X_kJLORAKS, mask] """
    
    def __init__(self, h5_path,augmentation=False,verbosity=False):
        self.fname = h5_path
        self.tables = tables.open_file(self.fname)
        self.nslices = self.tables.root.X_JLORAKS.shape[0]
        self.tables.close()
        self.X_JLORAKS = None # reconstructed images from JLORAKS
        self.Y_JLORAKS = None # Fully sampled images
        self.Sens = None # Sensitivity maps 
        self.X_kJLORAKS = None # Acquired K-space 
        self.augmentation = augmentation # Whether augment the data 
        self.verbose = verbosity # Print out what is going on?
        
    def AugmentFlip(self,im, axis):
        im = im.swapaxes(0,axis)
        im = im[::-1]
        im = im.swapaxes(axis,0)
        return im.copy()
    
    def AugmentScale(self,im,scale_val):
        im = im * scale_val
        return im
    def __getitem__(self, ind):
        
        if(self.X_JLORAKS is None): # Open in thread
            self.tables = tables.open_file(self.fname, 'r')
            self.X_JLORAKS = self.tables.root.X_JLORAKS
            self.Y_JLORAKS = self.tables.root.Y_JLORAKS
            self.Sens = self.tables.root.Sens
            self.X_kJLORAKS = self.tables.root.X_kJLORAKS
            
#        t0 = time.time()
        '''
        X_JLORAKS = np.float32(np.array(self.X_JLORAKS[ind]))
        Y_JLORAKS = np.float32(np.array(self.Y_JLORAKS[ind]))
        Sens = np.float32(np.array(self.Sens[ind]))
        X_kJLORAKS = np.float32(np.array(self.X_kJLORAKS[ind]))
        '''
        X_JLORAKS = np.float32(self.X_JLORAKS[ind])
        Y_JLORAKS = np.float32(self.Y_JLORAKS[ind])
        Sens = np.float32(self.Sens[ind])
        X_kJLORAKS = np.float32(self.X_kJLORAKS[ind])
#        '''
        
#        print(time.time()-t0)
        
        """ Data Loading """
        mask = np.float32(np.abs(X_kJLORAKS[:,0:1,0:1])>0) 
#        Sens = np.tile(Sens,[1,8,1,1,1])
        
        if self.verbose: 
            print('X_JLORAKS:',X_JLORAKS.shape, X_JLORAKS.dtype)
            print('Y_JLORAKS:', Y_JLORAKS.shape, Y_JLORAKS.dtype)
            print('Sens:', Sens.shape, Sens.dtype)
            print('X_kJLORAKS:', X_kJLORAKS.shape, X_kJLORAKS.dtype)
            print('mask:', mask.shape, mask.dtype)
        
        """ Augmentation (Random Flipping (None, Left-Right, Up-Down), Scaling (0.9 - 1.1) """
        if self.augmentation:
            """ Random Flipping """
            if np.random.random() < 0.5:
#                pdb.set_trace()
                X_JLORAKS = self.AugmentFlip(X_JLORAKS,1)
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS,1)
                Sens = self.AugmentFlip(Sens,2)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS,2)
                mask = self.AugmentFlip(mask,2)
                
            if np.random.random() < 0.5: 
                X_JLORAKS = self.AugmentFlip(X_JLORAKS,2)
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS,2)
                Sens = self.AugmentFlip(Sens,3)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS,3)
                mask = self.AugmentFlip(mask,3)
             
            scale_f = np.random.uniform(0.9,1.1)
            X_JLORAKS = self.AugmentScale(X_JLORAKS,scale_f)
            Y_JLORAKS = self.AugmentScale(Y_JLORAKS,scale_f)
            X_kJLORAKS = self.AugmentScale(X_kJLORAKS,scale_f)
                
        return torch.from_numpy(X_JLORAKS), torch.from_numpy(Y_JLORAKS),torch.from_numpy(Sens),torch.from_numpy(X_kJLORAKS), torch.from_numpy(mask)
    def __len__(self):
        return self.nslices

    
class MAGIC_Dataset_zpad(Dataset):
    """ Dataloader that Load & Augment & change to Torch Tensors (For Training)
        Inputs : path to Dataset (.h5)
        Outputs : [X_JLORAKS, Y_JLORAKS, Sens, X_kJLORAKS, mask] """
    
    def __init__(self, h5_path,augmentation=False,verbosity=False):
        self.fname = h5_path
        self.tables = tables.open_file(self.fname)
        self.nslices = self.tables.root.X_JLORAKS.shape[0]
        self.tables.close()
        self.Y_JLORAKS = None # Fully sampled images
        self.Sens = None # Sensitivity maps 
        self.augmentation = augmentation # Whether augment the data 
        self.verbose = verbosity # Print out what is going on?
        
    def AugmentFlip(self,im, axis):
        im = im.swapaxes(0,axis)
        im = im[::-1]
        im = im.swapaxes(axis,0)
        return im.copy()
    
    def AugmentScale(self,im,scale_val):
        im = im * scale_val
        return im
    def __getitem__(self, ind):
        
        if(self.Y_JLORAKS is None): # Open in thread
            self.tables = tables.open_file(self.fname, 'r')
            self.Y_JLORAKS = self.tables.root.Y_JLORAKS
            self.Sens = self.tables.root.Sens
            self.X_kJLORAKS = self.tables.root.X_kJLORAKS

        Y_JLORAKS = np.float32(self.Y_JLORAKS[ind])
        Sens = np.float32(self.Sens[ind])
        X_kJLORAKS = np.float32(self.X_kJLORAKS[ind])
        
        """ Data Loading """
        mask = np.float32(np.abs(X_kJLORAKS[:,0:1,0:1])>0) 

        """ Augmentation (Random Flipping (None, Left-Right, Up-Down), Scaling (0.9 - 1.1) """
        if self.augmentation:
            """ Random Flipping """
            if np.random.random() < 0.5:
#                pdb.set_trace()
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS,1)
                Sens = self.AugmentFlip(Sens,2)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS,2)
                mask = self.AugmentFlip(mask,2)
                
            if np.random.random() < 0.5: 
                Y_JLORAKS = self.AugmentFlip(Y_JLORAKS,2)
                Sens = self.AugmentFlip(Sens,3)
                X_kJLORAKS = self.AugmentFlip(X_kJLORAKS,3)
                mask = self.AugmentFlip(mask,3)
             
            scale_f = np.random.uniform(0.9,1.1)
            Y_JLORAKS = self.AugmentScale(Y_JLORAKS,scale_f)
            X_kJLORAKS = self.AugmentScale(X_kJLORAKS,scale_f)
            
        Y = torch.from_numpy(Y_JLORAKS)
        S = torch.from_numpy(Sens)
        X_k = torch.from_numpy(X_kJLORAKS)
        m = torch.from_numpy(mask)
        
        X = T.ifft2c(X_k)
        X = combine_all_coils(X,S,0)
        return X, Y, S, X_k, m
 
    def __len__(self):
        return self.nslices