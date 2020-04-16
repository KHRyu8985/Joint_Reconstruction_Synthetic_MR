import torch
import torch.nn as nn
import torch.nn.functional as F
import transforms as T
import data_loader as D

class MC_dataConsistencyTerm(nn.Module):
    """
    Inputs: 
    1. Coil Combined Image (x) : Slices, Contrast, XRes, YRes, (real, imag)
    2. Undersampled Kspace (k0) : Slices, Coils, Contrast, XRes, YRes, (real, imag)
    3. Mask (mask) : Slices, Coils, Contrast, XRes, YRes, (real, imag)
    4. Sensitivity maps (sensitivity): Slices, Coils, Contrast, XRes, YRes, (real, imag)
    
    Outputs:
    coil combined (out): Slices, Contrast, XRes, YRes, (real,imag)
    """
    def __init__(self, noise_lvl=None):
        super(MC_dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            noise_lvl_tensor = torch.Tensor([noise_lvl])*torch.ones(8) # Different lvl for each contrast & Channels            
            self.noise_lvl = torch.nn.Parameter(noise_lvl_tensor) 
            
        self.conv = nn.Sequential(
            ComplexConv2d(96, 96, 5, padding=2, bias=True)
            )


            
    def perform(self, x, k0, mask, sensitivity,coil_dim=1):
        k = T.fft2c(D.project_all_coils(x,sensitivity,coil_dim))
        k = k.view(k.size(0),k.size(1)*k.size(2),k.size(3),k.size(4),k.size(5))
        k = self.conv(k)
        k = k.view(k.size(0),12,8,k.size(2),k.size(3),k.size(4))
            
        if self.noise_lvl is not None: # noisy case
            v = torch.sigmoid(self.noise_lvl) # Normalize to 0~1
            v = v.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5)

            k = (1 - mask) * k + mask * (v * k + (1 - v) * k0) 
        else:  # noiseless case
            k = (1 - mask) * k + mask * k0
        return D.combine_all_coils(T.ifft2c(k),sensitivity,coil_dim)
    

class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            para = torch.Tensor([para])*torch.ones(8) # Different lvl for each contrast
            self.para = torch.nn.Parameter(para)
            
#            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx):
        para = torch.sigmoid(self.para) # Normalize to 0~1
        para = para.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4) 
        
        return para*cnn + (1 - para)*Sx

class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x): 
        return torch.stack((self.conv_re(x[..., 0]) - self.conv_im(x[..., 1]),\
                              self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])), dim=-1)
    
class MC_cnn_layer(nn.Module):
    """
    Inputs: Slices, Contrast, XRes, YRes, (real, imag) tensor
    
    Outputs: Slices, Contrast, XRes, YRes, (real, imag) Tensor (Denoised)
    """
    def __init__(self):
        super(MC_cnn_layer, self).__init__()
        
        self.conv = nn.Sequential(
            ComplexConv2d(8, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            ComplexConv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            ComplexConv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            ComplexConv2d(64, 8, 1, padding=0, bias=True)
        )     

    def forward(self, x):
        return self.conv(x)
    
class network(nn.Module):
    
    def __init__(self,  alfa=None, beta=1, cascades=5):
        super(network, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(MC_cnn_layer()) 
            dc_blocks.append(MC_dataConsistencyTerm(alfa)) 
            wa_blocks.append(weightedAverageTerm(beta)) 
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
         
    def forward(self, x, k, m, c):
                
        for i in range(self.cascades):
            Sx = self.dc_blocks[i].perform(x, k, m, c)
            x = self.conv_blocks[i](x) + x
            x = self.wa_blocks[i].perform(x, Sx)
        return x    

    