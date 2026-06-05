import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import sys
sys.path.append('../modules')
import model
try:
    from nvitop import Device
    NVITOP = True
except:
    NVITOP = False
from collections import OrderedDict
import pathlib
import matplotlib.pyplot as pl
import logging
import glob
import platform
import h5py
from einops import rearrange, repeat
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import av
import cv2
       
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):

        # Expand t to have the same batch size as x
        t = torch.zeros(x.shape[0], device=x.device) + t
        stokes = extras['stokes']
        return self.model(x, t, stokes)
        
class Testing(object):
    def __init__(self, checkpoint=None, gpu=0, batch_size=64):

        self.logger = logging.getLogger("flow ")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVITOP):            
            self.handle = Device.all()[0]
            memory_usage = f'{self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
            self.logger.info(f"Computing in {self.device} : {self.handle.name()} - Memory : {memory_usage}")
                
        self.batch_size = batch_size
                
        # Read checkpoiny
        if (checkpoint is None):
            files = glob.glob(f'../weights/*best.pth')
            self.checkpoint = max(files, key=os.path.getctime)
        else:
            self.checkpoint = '{0}'.format(checkpoint)
        
        self.logger.info("=> loading checkpoint '{}'".format(self.checkpoint))        
        checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage, weights_only=False)        
        self.logger.info("=> loaded checkpoint '{}'".format(self.checkpoint))

        self.hyperparameters = checkpoint['hyperparameters']
        self.normalization = checkpoint['normalization']        

        self.model_args_stokes = self.hyperparameters['unet_stokes']
        self.model_args = self.hyperparameters['unet_flow']
                
        self.model = model.ModelSimple(self.model_args, 
                           self.model_args_stokes, 
                           rank=0, 
                           handle=self.handle,
                           logger=self.logger).to(self.device)
                
        # Load the model state dict
        self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info('N. total parameters flow matching velocity model : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
                
        if 'precision' in self.hyperparameters['training']:
            if self.hyperparameters['training']['precision'] == 'fp16':
                self.use_amp = True
                self.logger.info('Using fp16 precision')
            else:
                self.use_amp = False
                self.logger.info('Using fp32 precision')
        else:
            self.use_amp = False

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.stokes_weight = torch.tensor(self.normalization['std_stokes'].astype('float32')).to(self.device)
        self.stokes_mean = torch.tensor(self.normalization['mn_stokes'].astype('float32')).to(self.device)

        self.variables = ['T', 'vz', 'tau', 'Bx', 'By', 'Bz', 'logP']
        self.abs_value = [False, False, False, True, False, False]  # whether to take absolute value or not to deal with sign ambiguity
        self.multiplication_factor = [1.0, 1e-5, 1.0, 1.0, 1.0, 1.0, 1.0]  # to convert to cgs where needed

    def test(self, n_pixel=32, noise=1e-3, originx=0, originy=0, zeroQUV=True):

        self.n_pixel = n_pixel
        self.model.eval()
        
        wrapped_vf = WrappedModel(self.model)

        step_size = 0.1

        f = h5py.File('../validation_stokes.h5', 'r')
        
        patch_stokes = f['stokes'][originx:originx+n_pixel, originy:originy+n_pixel, :, :].copy()
        patch_stokes = np.transpose(patch_stokes, (1, 0, 2, 3))
        patch_stokes = np.transpose(patch_stokes, axes=(2, 3, 0, 1))

        patch_stokes = torch.tensor(patch_stokes[None, :, :, :, :].astype('float32'), device=self.device)

        if zeroQUV:
            patch_stokes[:, 1:, :, :, :] *= 0.0
        
        patch_stokes_original = patch_stokes.clone()
        
        patch_stokes += noise * torch.randn_like(patch_stokes)

        patch_stokes_noise = patch_stokes.clone()

        patch_stokes -= self.stokes_mean[None, :, None, None, None]
        patch_stokes *= self.stokes_weight[None, :, None, None, None]
        patch_stokes = rearrange(patch_stokes, 'b s w x y -> b (s w) x y')

        patch_stokes = repeat(patch_stokes, '1 c x y -> b c x y', b=self.batch_size)

        f.close()

        f = h5py.File('../validation_model.h5', 'r')
        # f = h5py.File('/scratch1/datasets/spin4d_hinode/Hinode/MURaM/SPIN4D_SSD_Large/spin4d-ssdlarge-041689-Hinode.h5', 'r')

        patch_phys = []
        for k in range(len(self.variables)):
            var = self.multiplication_factor[k] * f[self.variables[k]][originx:originx+n_pixel, originy:originy+n_pixel, :].copy()
            # if self.abs_value[k]:
                # var = np.abs(var)
            patch_phys.append(var[:, :, None, :])

        Bx = patch_phys[3]
        By = patch_phys[4]
        Bt = np.sqrt(Bx**2 + By**2)  # horizontal field strength
        phi = np.atan2(By, Bx)  # azimuthal angle
        patch_phys[3] = Bt * np.cos(2 * phi)  # Bp1
        patch_phys[4] = Bt * np.sin(2 * phi)  # Bp2

        patch_phys[1] = -patch_phys[1]  # change vz sign to match observations convention
        
        patch_phys = np.concatenate(patch_phys, axis=2)

        f.close()

        with torch.no_grad():
            T = torch.tensor([0.0, 1.0], device=self.device)

            x_init = torch.randn((self.batch_size, 77, n_pixel, n_pixel), dtype=torch.float32, device=self.device)            
            solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class            
            sol = solver.sample(time_grid=T, 
                                x_init=x_init, 
                                stokes=patch_stokes, 
                                method='midpoint', 
                                step_size=step_size, 
                                return_intermediates=False)  # sample from the model
            
        sol = sol.cpu().numpy()
        
        sol *= self.normalization['std_models'].flatten()[None, :, None, None]
        sol += self.normalization['mn_models'].flatten()[None, :, None, None]

        sol = sol.reshape((self.batch_size, 7, 11, n_pixel, n_pixel))
        sol[:, 1, ...] *= -1.0  # change vz sign to match observations convention

        return sol, patch_phys, patch_stokes_noise.cpu().numpy(), patch_stokes_original.cpu().numpy()
                                        
if (__name__ == '__main__'):
    
    n_pixel = 64
    noise = 1e-3
    zeroQUV = False

    batch_size = 25
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5, -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variables = ['T [K]', 'vz [km/s]', 'h [km]', 'Bx [G]', 'By [G]', 'Bz [G]']    
                    
    deepnet = Testing(checkpoint=None, batch_size=batch_size)
    sol, phys, stokes_noise, stokes_original = deepnet.test(n_pixel=n_pixel, noise=noise, originx=0, originy=0, zeroQUV=zeroQUV)

    if zeroQUV:
        f = h5py.File('validation_results_zeropol.h5', 'w')
    else:
        f = h5py.File('validation_results.h5', 'w')
    f.create_dataset('sol', data=sol)
    f.create_dataset('phys', data=phys)
    f.close()