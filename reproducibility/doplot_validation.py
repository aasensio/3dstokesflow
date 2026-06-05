import torch
import numpy as np
import matplotlib.pyplot as pl
import h5py
import sys
sys.path.append('../util')
import interpolation
from mpl_toolkits.axes_grid1 import ImageGrid
import disambig
from einops import rearrange
import av 
import cv2
import torch.nn.functional as F
import witt
import dissipation
from tqdm import tqdm

pl.close('all')

def mad(x, axis=0):
    median = np.nanmedian(x, axis=axis)
    mad = np.nanmedian(np.abs(x - median), axis=axis)
    return mad

def plot_horizontal_validation(save=False, quantity=0):

    f = h5py.File('validation_results.h5', 'r')
    sol_stokes = f['sol'][:]
    phys = f['phys'][:]
    phys = np.transpose(phys, (2, 3, 0, 1))  # [B, H, X, Y]
    f.close()
    
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5,  -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variable_names = ['T', 'v_z', 'h', 'B_p1', 'B_p2', 'B_z', 'logP']
    variables = ['T', r'v$_z$', 'h', r'B$_p1$', r'B$_p2$', r'B$_z$', 'logP'] 
    units = ['K', 'km/s', 'km', 'G', 'G', 'G', 'dex']

    # Define the z axis
    z = np.array([-50.0, 100.0, 200.0, 300.0, 400.0])
    z = torch.tensor(z, dtype=torch.float32)

    # Height axes for each solution
    h = phys[2, ::-1, ...]
    z0_phys = np.mean(h[2, ...])
    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...][:, ::-1, :, :].copy()
    T = sol_stokes[:, quantity, ...]
    z0_stokes = np.mean(h[:, 2, :, :])
    
    # Interpolation
    T = torch.tensor(T, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    out_stokes = interpolation.interpolate_heights(T, h - z0_stokes, z, fill_value=np.nan)

    ##############
    # Original data
    ##############
    h = phys[2, ::-1, ...][None, ...].copy()
    T = phys[quantity, :, ...][None, ...]

    T = torch.tensor(T, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    out_phys = interpolation.interpolate_heights(T, h - z0_phys, z, fill_value=np.nan)
    
    if quantity == 0:
        minval = [5000, 4000, 4000, 4000, 4000]
        maxval = [10000, 6500, 6500, 6500, 6000]
        cmap = 'inferno'
        label_color = ['white', 'white']

    if quantity == 1:
        minval = [-3, -3, -3, -5, -5]
        maxval = [3, 3, 3, 5, 5]
        cmap = 'RdBu_r'
        label_color = ['black', 'black']

    if quantity == 5:
        minval = [-500, -500, -300, -300, -300]
        maxval = [500, 500, 300, 300, 300]
        cmap = 'RdBu_r'
        label_color = ['white', 'white']

    if quantity == 6:
        minval = [3.3, 3.3, 3.3, 3.3, 3.3]
        maxval = [5.2, 5.2, 5.2, 5.2, 5.2]
        cmap = 'RdBu_r'
        label_color = ['black', 'black']

    low = 2
    up = -2

    nx, ny = out_stokes[:, :, low:up, low:up].shape[2], out_stokes[:, :, low:up, low:up].shape[3]
    labels = ['Median', 'MAD (x10)', 'Sim']
    fig, ax = pl.subplots(nrows=5, ncols=3, figsize=(16.5, 25), layout='constrained', sharex=True, sharey=True)

    cmap_obj = pl.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='black')  # masked values will be black

    for i in range(5):
        data_stokes = np.nanmedian(out_stokes[:, i, low:up, low:up].cpu().numpy(), axis=0)
        data_stokes_mad = mad(out_stokes[:, i, low:up, low:up].cpu().numpy(), axis=0)
        data_phys = out_phys[0, i, low:up, low:up].cpu().numpy()

        data_stokes = np.ma.masked_where(data_stokes == np.nan, data_stokes)
        data_phys = np.ma.masked_where(data_phys == np.nan, data_phys)

        im = ax[i, 0].imshow(
            data_stokes,
            vmin=minval[i],
            vmax=maxval[i],
            extent=[0, ny * 0.16, 0, nx * 0.16],
            cmap=cmap_obj,
            origin='lower'
        )
        im = ax[i, 1].imshow(
            data_stokes_mad*10,
            vmin=0.0,
            vmax=maxval[i],
            extent=[0, ny * 0.16, 0, nx * 0.16],
            cmap=cmap_obj,
            origin='lower'
        )
        im = ax[i, 2].imshow(
            data_phys,
            vmin=minval[i],
            vmax=maxval[i],
            extent=[0, ny * 0.16, 0, nx * 0.16],
            cmap=cmap_obj,
            origin='lower'
        )

        cbar = fig.colorbar(im, ax=ax[i, :], location='right', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(f'{variables[quantity]} [{units[quantity]}]', fontsize=14)

        ax[i, 0].text(
            0.05, 0.1, f'{z[i]:.0f} km',
            transform=ax[i, 0].transAxes,
            fontsize=18,
            verticalalignment='top',
            color=label_color[0],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

    for i in range(3):
        ax[0, i].text(
            0.05, 0.95, f'{labels[i]}',
            transform=ax[0, i].transAxes,
            fontsize=18,
            verticalalignment='top',
            color=label_color[1],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)

    if save:
        pl.savefig(f'figs/validation_{variable_names[quantity]}.pdf', dpi=300)

def plot_horizontal_samples(save=False, quantity=0):

    f = h5py.File('validation_results.h5', 'r')
    sol_stokes = f['sol'][:]
    f.close()
    
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5,  -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variable_names = ['T', 'v_z', 'h', 'B_p1', 'B_p2', 'B_z', 'log P']
    variables = ['T', r'v$_z$', 'h', r'B$_p1$', r'B$_p2$', r'B$_z$', 'log P'] 
    units = ['K', 'km/s', 'km', 'G', 'G', 'G', 'dex']

    # Define the z axis
    z = np.array([-50.0, 100.0, 200.0, 300.0, 400.0])
    z = torch.tensor(z, dtype=torch.float32)
    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...][:, ::-1, :, :].copy()
    T = sol_stokes[:, quantity, ...]
    z0_stokes = np.mean(h[:, 2, :, :])
    
    # Interpolation
    T = torch.tensor(T, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    out_stokes = interpolation.interpolate_heights(T, h - z0_stokes, z, fill_value=10000.0)
    
    if quantity == 0:
        minval = [5000, 4000, 4000, 4000, 4000]
        maxval = [10000, 6500, 6500, 6500, 6000]
        cmap = 'inferno'
        label_color = ['white', 'white']

    if quantity == 1:
        minval = [-3, -3, -3, -3, -3]
        maxval = [3, 3, 3, 3, 3]
        cmap = 'RdBu_r'
        label_color = ['white', 'white']

    if quantity == 5 or quantity == 3:
        minval = [-500, -500, -300, -300, -300]
        maxval = [500, 500, 300, 300, 300]
        cmap = 'RdBu_r'
        label_color = ['black', 'black']

    if quantity == 6:
        minval = [3.3, 3.3, 3.3, 3.3, 3.3]
        maxval = [5.2, 5.2, 5.2, 5.2, 5.2]
        cmap = 'RdBu_r'
        label_color = ['black', 'black']

    low = 2
    up = -2

    nx, ny = out_stokes[:, :, low:up, low:up].shape[2], out_stokes[:, :, low:up, low:up].shape[3]
    labels = ['IQUV', 'Sim']
    fig, ax = pl.subplots(nrows=5, ncols=6, figsize=(18.0, 15), layout='constrained', sharex=True, sharey=True)

    cmap_obj = pl.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='black')  # masked values will be black

    for i in range(5):
        for j in range(6):
            data_stokes = out_stokes[j, i, low:up, low:up].cpu().numpy()
            data_stokes = np.ma.masked_where(data_stokes == 10000.0, data_stokes)
            im = ax[i, j].imshow(data_stokes, vmin=minval[i], vmax=maxval[i], extent=[0, ny*0.16, 0, nx*0.16], cmap=cmap_obj, origin='lower')
        
        cbar = fig.colorbar(im, ax=ax[i, :], location='right', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(f'{variables[quantity]} [{units[quantity]}]', fontsize=14)

    for i in range(6):
        ax[0, i].set_title(f'Sample {i}', fontsize=15)

    for i in range(5):
        ax[i, 0].text(
            0.05, 0.12, f'{z[i]:.0f} km',
            transform=ax[i, 0].transAxes,
            fontsize=18,
            verticalalignment='top',
            color=label_color[0],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

    # for i in range(2):
    #     ax[0, i].text(
    #         0.05, 0.95, f'{labels[i]}',
    #         transform=ax[0, i].transAxes,
    #         fontsize=18,
    #         verticalalignment='top',
    #         color=label_color[1],
    #         fontweight='bold'
    #     )

    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)

    if save:
        pl.savefig(f'figs/samples_{variable_names[quantity]}.pdf', dpi=300)

def calculate_div(save=False):
    
    f = h5py.File('validation_results.h5', 'r')
    sol_stokes = f['sol'][:]
    phys = f['phys'][:]
    phys = np.transpose(phys, (2, 3, 0, 1))  # [B, H, X, Y]
    f.close()
    
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5,  -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variable_names = ['T', 'v_z', 'h', 'B_p1', 'B_p2', 'B_z']
    variables = ['T', r'v$_z$', 'h', r'B$_p1$', r'B$_p2$', r'B$_z$'] 
    units = ['K', 'km/s', 'km', 'G', 'G', 'G']

    deltax = 0.16 * 725.0
    deltay = 0.16 * 725.0
    deltaz = 10.0

    # Define the z axis
    z = np.array([90.0, 100.0])
    z = torch.tensor(z, dtype=torch.float32)

    # Height axes for each solution
    h = phys[2, :, ...]   
    z0_phys = np.mean(h[8, ...])
    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...][:, ::-1, :, :].copy()
    Bp1 = sol_stokes[:, 3, ...].copy()
    Bp2 = sol_stokes[:, 4, ...].copy()
    Bz = sol_stokes[:, 5, ...].copy()
    z0_stokes = np.mean(h[:, 2, :, :])
    
    # Interpolation
    Bp1 = torch.tensor(Bp1, dtype=torch.float32)
    Bp2 = torch.tensor(Bp2, dtype=torch.float32)
    Bz = torch.tensor(Bz, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    Bp1_stokes = interpolation.interpolate_heights(Bp1, h - z0_stokes, z, fill_value=0.0)
    Bp2_stokes = interpolation.interpolate_heights(Bp2, h - z0_stokes, z, fill_value=0.0)
    Bz_stokes = interpolation.interpolate_heights(Bz, h - z0_stokes, z, fill_value=0.0)
    
    BT = torch.sqrt(Bp1_stokes**2 + Bp2_stokes**2)
    twophi = torch.atan2(Bp2_stokes, Bp1_stokes)
    phi = (0.5 * twophi) % torch.pi
    Bx_stokes = BT * torch.cos(phi)
    By_stokes = BT * torch.sin(phi)

    ##############
    # Original data
    ##############
    f = h5py.File('../validation_model.h5', 'r')
    h = np.transpose(f['tau'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    Bx = np.transpose(f['Bx'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    By = np.transpose(f['By'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    Bz = np.transpose(f['Bz'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    f.close()
    
    Bx = torch.tensor(Bx, dtype=torch.float32)
    By = torch.tensor(By, dtype=torch.float32)
    Bz = torch.tensor(Bz, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    Bx_phys = interpolation.interpolate_heights(Bx, h - z0_phys, z, fill_value=0.0)
    By_phys = interpolation.interpolate_heights(By, h - z0_phys, z, fill_value=0.0)
    Bz_phys = interpolation.interpolate_heights(Bz, h - z0_phys, z, fill_value=0.0)
    Bt = torch.sqrt(Bx_phys**2 + By_phys**2)  # horizontal field strength
    phi = torch.atan2(By_phys, Bx_phys)  # azimuthal angle
    Bp1_phys = Bt * torch.cos(2 * phi)  # Bp1
    Bp2_phys = Bt * torch.sin(2 * phi)  # Bp2
            
    model = disambig.DifferentiableME0(Bx_stokes, 
                                       By_stokes, 
                                       Bz_stokes,                                       
                                       dx=deltax, 
                                       dy=deltay,
                                       dz=deltaz, 
                                       device='cuda')
    
    Bx, By, signs, divB, (Jx, Jy, Jz) = model.optimize(epochs=5000, border=1, lr=0.01)

    np.savez('validation_div.npz', divB=divB.cpu().numpy(), Jx=Jx.cpu().numpy(), Jy=Jy.cpu().numpy(), Jz=Jz.cpu().numpy(), Bx=Bx.cpu().numpy(), By=By.cpu().numpy())

def plot_div(save=False):

    f = h5py.File('validation_results.h5', 'r')
    sol_stokes = f['sol'][:]
    phys = f['phys'][:]
    phys = np.transpose(phys, (2, 3, 0, 1))  # [B, H, X, Y]
    f.close()
    
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5,  -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variable_names = ['T', 'v_z', 'h', 'B_p1', 'B_p2', 'B_z']
    variables = ['T', r'v$_z$', 'h', r'B$_p1$', r'B$_p2$', r'B$_z$'] 
    units = ['K', 'km/s', 'km', 'G', 'G', 'G']

    deltax = 0.16 * 725.0
    deltay = 0.16 * 725.0
    deltaz = 10.0

    # Define the z axis
    z = np.array([90.0, 100.0])
    z = torch.tensor(z, dtype=torch.float32)

    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...][:, ::-1, :, :].copy()
    Bp1 = sol_stokes[:, 3, ...].copy()
    Bp2 = sol_stokes[:, 4, ...].copy()
    Bz = sol_stokes[:, 5, ...].copy()
    z0_stokes = np.mean(h[:, 2, :, :])
    
    # Interpolation
    Bp1 = torch.tensor(Bp1, dtype=torch.float32)
    Bp2 = torch.tensor(Bp2, dtype=torch.float32)
    Bz = torch.tensor(Bz, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    Bp1_stokes = interpolation.interpolate_heights(Bp1, h - z0_stokes, z, fill_value=0.0)
    Bp2_stokes = interpolation.interpolate_heights(Bp2, h - z0_stokes, z, fill_value=0.0)
    Bz_stokes = interpolation.interpolate_heights(Bz, h - z0_stokes, z, fill_value=0.0)
    
    BT = torch.sqrt(Bp1_stokes**2 + Bp2_stokes**2)
    twophi = torch.atan2(Bp2_stokes, Bp1_stokes)
    phi = (0.5 * twophi) % torch.pi
    Bx_stokes = BT * torch.cos(phi)
    By_stokes = BT * torch.sin(phi)

    ##############
    # Original data
    ##############
    f = h5py.File('../validation_model.h5', 'r')        
    h = np.transpose(f['tau'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    Bx = np.transpose(f['Bx'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    By = np.transpose(f['By'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    Bz = np.transpose(f['Bz'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    f.close()

    h = h[:, ::-1, :, :].copy()

    z0_phys = np.mean(h[:, 2, ...])
    
    Bx = torch.tensor(Bx, dtype=torch.float32)
    By = torch.tensor(By, dtype=torch.float32)
    Bz = torch.tensor(Bz, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    Bx_phys = interpolation.interpolate_heights(Bx, h - z0_phys, z, fill_value=0.0)
    By_phys = interpolation.interpolate_heights(By, h - z0_phys, z, fill_value=0.0)
    Bz_phys = interpolation.interpolate_heights(Bz, h - z0_phys, z, fill_value=0.0)    
    Bt = torch.sqrt(Bx_phys**2 + By_phys**2)  # horizontal field strength
    phi = torch.atan2(By_phys, Bx_phys)  # azimuthal angle
    Bp1_phys = Bt * torch.cos(2 * phi)  # Bp1
    Bp2_phys = Bt * torch.sin(2 * phi)  # Bp2

    Jx_phys, Jy_phys, Jz_phys = disambig.compute_J(Bx_phys, By_phys, Bz_phys, deltax, deltay, deltaz)
    Jx_phys = Jx_phys[0, ...]
    Jy_phys = Jy_phys[0, ...]
    Jz_phys = Jz_phys[0, ...]
    
    data = np.load('validation_div.npz')
    divB = data['divB']
    Jx = data['Jx']
    Jy = data['Jy']
    Jz = data['Jz']
    Bx = data['Bx']
    By = data['By']

    low = 2
    up = -2


    ##########################################
    # Bx/By
    ##########################################

    nx, ny = Bp1_stokes[0, 0, low:up, low:up].shape[0], Bp1_stokes[0, 0, low:up, low:up].shape[1]

    
    fig, ax = pl.subplots(nrows=3, ncols=4, figsize=(25, 16), layout='constrained', sharex=True, sharey=True)
    im = ax[0, 0].imshow(np.median(Bp1_stokes[:, 0, low:up, low:up], axis=0), vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 0].imshow(mad(Bp1_stokes[:, 0, low:up, low:up], axis=0), vmin=0, vmax=60, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[2, 0].imshow(Bp1_phys[0, 0, low:up, low:up], vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 0], location='right', fraction=0.046, pad=0.04)
    ax[0, 0].set_title(r'B$_{p1}$ [G]', fontsize=16)
        
    im = ax[0, 1].imshow(np.median(Bp2_stokes[:, 0, low:up, low:up], axis=0), vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(mad(Bp2_stokes[:, 0, low:up, low:up], axis=0), vmin=0, vmax=60, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[2, 1].imshow(Bp2_phys[0, 0, low:up, low:up], vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 1], location='right', fraction=0.046, pad=0.04)
    ax[0, 1].set_title(r'B$_{p2}$ [G]', fontsize=16)
    
    im = ax[0, 2].imshow(np.median(Bx[:, low:up, low:up], axis=0), vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 2], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 2].imshow(mad(Bx[:, low:up, low:up], axis=0), vmin=0, vmax=60, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 2], location='right', fraction=0.046, pad=0.04)
    im = ax[2, 2].imshow(Bx_phys[0, 0, low:up, low:up], vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 2], location='right', fraction=0.046, pad=0.04)
    ax[0, 2].set_title(r'B$_x$ [G]', fontsize=16)
    
    
    im = ax[0, 3].imshow(np.median(By[:, low:up, low:up], axis=0), vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 3], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 3].imshow(mad(By[:, low:up, low:up], axis=0), vmin=0, vmax=60, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 3], location='right', fraction=0.046, pad=0.04)
    im = ax[2, 3].imshow(By_phys[0, 0, low:up, low:up], vmin=-200, vmax=200, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 3], location='right', fraction=0.046, pad=0.04)
    ax[0, 3].set_title(r'B$_y$ [G]', fontsize=16)

    ax[0, 0].text(
        0.05, 0.95, f'IQUV Median',
        transform=ax[0, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='black',
        fontweight='bold'
    )

    ax[1, 0].text(
        0.05, 0.95, f'IQUV MAD',
        transform=ax[1, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='white',
        fontweight='bold'
    )

    ax[2, 0].text(
        0.05, 0.95, f'Sim',
        transform=ax[2, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='black',
        fontweight='bold'
    )
    
    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)
    
    if save:
        pl.savefig(f'figs/validation_Bxyz.pdf', dpi=300)

    
    ##########################################
    # CURRENTS
    ##########################################
    
    nx, ny = Jx[:, low:up, low:up].shape[1], Jx[:, low:up, low:up].shape[2]

    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(21, 18), layout='constrained', sharex=True, sharey=True)
    im = ax[0, 0].imshow(np.median(Jx[:, low:up, low:up], axis=0), vmin=-0.1, vmax=0.1, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(np.median(Jy[:, low:up, low:up], axis=0), vmin=-0.1, vmax=0.1, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[0, 2].imshow(np.median(Jz[:, low:up, low:up], axis=0), vmin=-0.05, vmax=0.05, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 2], location='right', fraction=0.046, pad=0.04)
    ax[0, 0].set_title(r'J$_x$ [A/m$^2$]', fontsize=16)
    ax[0, 1].set_title(r'J$_y$ [A/m$^2$]', fontsize=16)
    ax[0, 2].set_title(r'J$_z$ [A/m$^2$]', fontsize=16)

    im = ax[1, 0].imshow(mad(Jx[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.05, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(mad(Jy[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.05, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 2].imshow(mad(Jz[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.02, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 2], location='right', fraction=0.046, pad=0.04)

    im = ax[2, 0].imshow(Jx_phys[low:up, low:up], vmin=-0.1, vmax=0.1, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[2, 1].imshow(Jy_phys[low:up, low:up], vmin=-0.1, vmax=0.1, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[2, 2].imshow(Jz_phys[low:up, low:up], vmin=-0.05, vmax=0.05, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 2], location='right', fraction=0.046, pad=0.04)

    ax[0, 0].text(
        0.05, 0.95, f'IQUV Median',
        transform=ax[0, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='black',
        fontweight='bold'
    )

    ax[1, 0].text(
        0.05, 0.95, f'IQUV MAD',
        transform=ax[1, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='white',
        fontweight='bold'
    )

    ax[2, 0].text(
        0.05, 0.95, f'Sim',
        transform=ax[2, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='black',
        fontweight='bold'
    )

    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)

    if save:
        pl.savefig(f'figs/validation_currents.pdf', dpi=300)


    ##########################################
    # DISSIPATION
    ##########################################

    data = np.load('quantities_for_dissipation.npz')
    Pe_phys = data['Pe_phys']
    Pgas_phys = data['Pgas_phys']
    Pe = data['Pe']
    Pgas = data['Pgas']
    temp_phys = data['temp_phys']
    temp = data['temp']
    Jx_phys = data['Jx_phys']
    Jy_phys = data['Jy_phys']
    Jz_phys = data['Jz_phys']
    Jx = data['Jx']
    Jy = data['Jy']
    Jz = data['Jz']
    Bx_phys = data['Bx_phys']
    By_phys = data['By_phys']
    Bz_phys = data['Bz_phys']
    Bx = data['Bx']
    By = data['By']
    Bz = data['Bz']

    BK = 1.380649e-16  # Boltzmann constant in erg/K

    Na = (Pgas_phys - Pe_phys) / (BK * temp_phys)
    Ne = Pe_phys          / (BK * temp_phys)
    Ni = Ne
            
    Q_ohm_phys, Q_amb_phys = dissipation.compute_heating_rates(Jx_phys, Jy_phys, Jz_phys, Bx_phys, By_phys, Bz_phys, temp_phys, Ne, Ni, Na)

    Na = (Pgas - Pe) / (BK * temp)
    Ne = Pe          / (BK * temp)
    Ni = Ne
        
    Q_ohm, Q_amb = dissipation.compute_heating_rates(Jx, Jy, Jz, Bx, By, Bz, temp, Ne, Ni, Na)

    fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10, 13), layout='constrained', sharex=True, sharey=True)
    im = ax[0, 0].imshow(np.median(Q_ohm[:, low:up, low:up], axis=0), vmin=0., vmax=0.01, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(np.median(Q_amb[:, low:up, low:up], axis=0), vmin=0., vmax=0.001, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 1], location='right', fraction=0.046, pad=0.04)    
    ax[0, 0].set_title(r'Q$_{ohm}$ [erg cm$^{-3}$ s$^{-1}$]', fontsize=16)
    ax[0, 1].set_title(r'Q$_{amb}$ [erg cm$^{-3}$ s$^{-1}$]', fontsize=16)
    
    im = ax[1, 0].imshow(mad(Q_ohm[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.001, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(mad(Q_amb[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.0001, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 1], location='right', fraction=0.046, pad=0.04)

    im = ax[2, 0].imshow(Q_ohm_phys[low:up, low:up], vmin=0.0, vmax=0.01, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[2, 1].imshow(Q_amb_phys[low:up, low:up], vmin=0.0, vmax=0.001, extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[2, 1], location='right', fraction=0.046, pad=0.04)
    
    ax[0, 0].text(
        0.05, 0.95, f'IQUV Median',
        transform=ax[0, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='white',
        fontweight='bold'
    )

    ax[1, 0].text(
        0.05, 0.95, f'IQUV MAD',
        transform=ax[1, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='white',
        fontweight='bold'
    )

    ax[2, 0].text(
        0.05, 0.95, f'Sim',
        transform=ax[2, 0].transAxes,
        fontsize=22,
        verticalalignment='top',
        color='white',
        fontweight='bold'
    )
    
    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)

    if save:
        pl.savefig(f'figs/validation_dissipation.pdf', dpi=300)


def plot_vertical_validation(save=False, quantity=0):

    f = h5py.File('validation_results.h5', 'r')
    sol_stokes = f['sol'][:]
    phys = f['phys'][:]
    phys = np.transpose(phys, (2, 3, 0, 1))  # [B, H, X, Y]
    f.close()
    
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5,  -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variable_names = ['T', 'v_z', 'h', 'B_x', 'B_y', 'B_z']
    variables = ['T', r'v$_z$', 'h', r'B$_x$', r'B$_y$', r'B$_z$'] 
    units = ['K', 'km/s', 'km', 'G', 'G', 'G']

    # Define the z axis
    z = np.array([-50.0, 100.0, 200.0, 300.0, 400.0])
    z = torch.tensor(z, dtype=torch.float32)

    # Height axes for each solution
    h = phys[2, :, ...]   
    z0_phys = np.mean(h[8, ...])
    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...]
    T = sol_stokes[:, quantity, ...]
    z0_stokes = np.mean(h[:, 8, :, :])
    
    # Interpolation
    T = torch.tensor(T, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    out_stokes = interpolation.interpolate_heights(T, h - z0_stokes, z, fill_value=0.0)

    ##############
    # Original data
    ##############
    h = phys[2, :, ...][None, ...]
    T = phys[quantity, :, ...][None, ...]

    T = torch.tensor(T, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    out_phys = interpolation.interpolate_heights(T, h - z0_phys, z, fill_value=0.0)
    
    fig, ax = pl.subplots(nrows=3, ncols=4, figsize=(22.0, 12), layout='constrained', sharex='col', sharey=False)

    indx = [38, 44, 22]
    indy = [6, 51, 56]
    quantities = [0, 1, 5]
    range_from = [3500, -8, -150]
    ranges_to = [11000, 8, 150]

    f = h5py.File('../validation_stokes.h5', 'r')

    patch = f['stokes'][:, :, :, :].copy()

    stokesI = patch[:, :, 0, 0] / np.mean(patch[:, :, 0, 0])
    stokesQ = patch[:, :, 1, 20] / np.mean(patch[:, :, 0, 0])
    stokesU = patch[:, :, 2, 20] / np.mean(patch[:, :, 0, 0])
    stokesV = patch[:, :, 3, 20] / np.mean(patch[:, :, 0, 0])

    im = ax[0, 0].imshow(stokesI, extent=[0, 64*0.16, 0, 64*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[0, 0], location='right', fraction=0.046, pad=0.04).set_label(r'I/I$_c$', fontsize=16)

    for i in range(3):
        for j in range(3):
            ax[j, 0].plot(indy[i]*0.16, indx[i]*0.16, marker='*', markersize=12, color=f'C{i}', markeredgecolor=f'C{i}', markeredgewidth=1.5)
        print(stokesI[indx[i], indy[i]])


    im = ax[1, 0].imshow(100*np.sqrt(stokesQ**2 + stokesU**2), extent=[0, 64*0.16, 0, 64*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 0], location='right', fraction=0.046, pad=0.04).set_label(r'L/I$_c$ [%]', fontsize=16)
    
    im = ax[2, 0].imshow(100*stokesV, extent=[0, 64*0.16, 0, 64*0.16], cmap='RdBu_r', vmin=-2, vmax=2, origin='lower')
    pl.colorbar(im, ax=ax[2, 0], location='right', fraction=0.046, pad=0.04).set_label(r'V/I$_c$ [%]', fontsize=16)
    
    
    for j in range(3):
        quantity = quantities[j]
        for i in range(3):
        
            # pct = np.percentile(sol_Ic[:, quantity, :, indx[i], indy[i]], [5, 50, 95], axis=0)
            # ax[i, j+1].plot(tau, pct[0, :], color='C1', linewidth=2, label='Zero QUV')
            # ax[i, j+1].plot(tau, pct[1, :], color='C1', linewidth=1)
            # ax[i, j+1].plot(tau, pct[2, :], color='C1', linewidth=2)
            # ax[i, j+1].fill_between(tau, pct[0, :], pct[2, :], color='C1', alpha=0.2)

            pct = np.percentile(sol_stokes[:, quantity, :, indx[i], indy[i]], [5, 25, 50, 75, 95], axis=0)
            ax[i, j+1].plot(tau, pct[0, :], color='C0', linewidth=2, label='IQUV')
            ax[i, j+1].plot(tau, pct[1, :], color='C0', linewidth=2)
            ax[i, j+1].plot(tau, pct[2, :], color='C0', linewidth=1)
            ax[i, j+1].plot(tau, pct[3, :], color='C0', linewidth=2)
            ax[i, j+1].plot(tau, pct[4, :], color='C0', linewidth=2)
            
            ax[i, j+1].fill_between(tau, pct[0, :], pct[4, :], color='C0', alpha=0.2)
            ax[i, j+1].fill_between(tau, pct[1, :], pct[3, :], color='C0', alpha=0.4)
    
            ax[i, j+1].plot(tau, phys[quantity, :, indx[i], indy[i]], color='C1', label='Sim', linewidth=2)

            ax[i, j+1].set_ylim(range_from[j], ranges_to[j])

        
    for i in range(3):
        ax[i, 1].plot(1, 4000, marker='*', markersize=12, color=f'C{i}', markeredgecolor=f'C{i}', markeredgewidth=1.5)

    for i in range(3):
        ax[i, 0].set_ylabel(f'Y [arcsec]', fontsize=18)
        ax[-1, i+1].set_xlabel(r'log $\tau$', fontsize=18)
        ax[i, 1].set_ylabel('T [K]', fontsize=18)
        ax[i, 2].set_ylabel(r'v$_z$ [km/s]', fontsize=18)
        ax[i, 3].set_ylabel('B$_z$ [G]', fontsize=18)
    ax[-1, 0].set_xlabel(f'X [arcsec]', fontsize=18)

    ax[0, 1].legend(loc='upper left', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)


    if save:
        pl.savefig(f'figs/vertical_validation.pdf', dpi=300)
        

def calculate_dissipation():

    f = h5py.File('validation_results.h5', 'r')
    sol_stokes = f['sol'][:]        
    f.close()
    
    deltax = 0.16 * 725.0
    deltay = 0.16 * 725.0
    deltaz = 10.0

    # Define the z axis
    z = np.array([90.0, 100.0])
    z = torch.tensor(z, dtype=torch.float32)
    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...][:, ::-1, :, :].copy()
    T = sol_stokes[:, 0, ...].copy()
    logP = sol_stokes[:, 6, ...].copy()    
    Bz = sol_stokes[:, 5, ...].copy()
    z0_stokes = np.mean(h[:, 2, :, :])
    
    # Interpolation    
    Bz = torch.tensor(Bz, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.float32)
    logP = torch.tensor(logP, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)    
    Bz = interpolation.interpolate_heights(Bz, h - z0_stokes, z, fill_value=0.0)
    T = interpolation.interpolate_heights(T, h - z0_stokes, z, fill_value=0.0)
    logP = interpolation.interpolate_heights(logP, h - z0_stokes, z, fill_value=0.0)

    data = np.load('validation_div.npz')
    divB = data['divB']
    Jx = data['Jx'] * 3e5
    Jy = data['Jy'] * 3e5
    Jz = data['Jz'] * 3e5
    Bx = data['Bx']
    By = data['By']    
    Bz = Bz[:, 0, ...].cpu().numpy()
    
    ##############
    # Original data
    ##############
    f = h5py.File('../validation_model.h5', 'r')
    h_phys = np.transpose(f['tau'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    T_phys = np.transpose(f['T'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    logP_phys = np.transpose(f['logP'][:][None, :, :, :], (0, 3, 1, 2))
    Bx_phys = np.transpose(f['Bx'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    By_phys = np.transpose(f['By'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    Bz_phys = np.transpose(f['Bz'][:][None, :, :, :], (0, 3, 1, 2))  # [B, H, X, Y]
    f.close()

    h_phys = h_phys[:, ::-1, :, :].copy()

    z0_phys = np.mean(h_phys[:, 2, ...])
    
    Bx_phys = torch.tensor(Bx_phys, dtype=torch.float32)
    By_phys = torch.tensor(By_phys, dtype=torch.float32)
    Bz_phys = torch.tensor(Bz_phys, dtype=torch.float32)
    T_phys = torch.tensor(T_phys, dtype=torch.float32)
    logP_phys = torch.tensor(logP_phys, dtype=torch.float32)
    h_phys = torch.tensor(h_phys, dtype=torch.float32)
    Bx_phys = interpolation.interpolate_heights(Bx_phys, h_phys - z0_phys, z, fill_value=0.0)
    By_phys = interpolation.interpolate_heights(By_phys, h_phys - z0_phys, z, fill_value=0.0)
    Bz_phys = interpolation.interpolate_heights(Bz_phys, h_phys - z0_phys, z, fill_value=0.0)    
    T_phys = interpolation.interpolate_heights(T_phys, h_phys - z0_phys, z, fill_value=0.0)
    logP_phys = interpolation.interpolate_heights(logP_phys, h_phys - z0_phys, z, fill_value=0.0)
    
    Jx_phys, Jy_phys, Jz_phys = disambig.compute_J(Bx_phys, By_phys, Bz_phys, deltax, deltay, deltaz)
    Jx_phys = Jx_phys[0, ...] * 3e5
    Jy_phys = Jy_phys[0, ...] * 3e5
    Jz_phys = Jz_phys[0, ...] * 3e5
    
    eos = witt.witt()
    
    # Simulation
    Pe_phys = np.zeros((64, 64))
    Pgas_phys = 10.0**logP_phys[0, 0, :, :].cpu().numpy()
    temp_phys = T_phys[0, 0, :, :].cpu().numpy()
    Jx_phys = Jx_phys[:, :].cpu().numpy()
    Jy_phys = Jy_phys[:, :].cpu().numpy()
    Jz_phys = Jz_phys[:, :].cpu().numpy()
    Bx_phys = Bx_phys[0, 0, :, :].cpu().numpy()
    By_phys = By_phys[0, 0, :, :].cpu().numpy()
    Bz_phys = Bz_phys[0, 0, :, :].cpu().numpy()
    

    # Inversion
    Pe = np.zeros((25, 64, 64))
    Pgas = 10.0**logP[:, 0, :, :].cpu().numpy()
    temp = T[:, 0, :, :].cpu().numpy()
    Jx = Jx[:, :, :]
    Jy = Jy[:, :, :]
    Jz = Jz[:, :, :]
    Bx = Bx[:, :, :]
    By = By[:, :, :]
    Bz = Bz[:, :, :]
    
    for i in tqdm(range(64)):
        for j in range(64):
            if temp_phys[i, j] != 0:
                Pe_phys[i, j] = eos.pe_from_pg(temp_phys[i, j], Pgas_phys[i, j])
            for k in range(25):
                if temp[k, i, j] != 0:
                    Pe[k, i, j] = eos.pe_from_pg(temp[k, i, j], Pgas[k, i, j])

    np.savez('quantities_for_dissipation.npz',
             Pe_phys=Pe_phys,
             Pgas_phys=Pgas_phys,
             Pe=Pe,
             Pgas=Pgas,
             temp_phys=temp_phys,
             temp=temp,
             Jx_phys=Jx_phys,
             Jy_phys=Jy_phys,
             Jz_phys=Jz_phys,
             Jx=Jx,
             Jy=Jy,
             Jz=Jz,
             Bx_phys=Bx_phys,
             By_phys=By_phys,
             Bz_phys=Bz_phys,
             Bx=Bx,
             By=By,
             Bz=Bz)


def plot_lorentz(save=False):
    tmp = np.load('quantities_for_dissipation.npz')
    Jx_phys = tmp['Jx_phys']
    Jy_phys = tmp['Jy_phys']
    Jz_phys = tmp['Jz_phys']
    Bx_phys = tmp['Bx_phys']
    By_phys = tmp['By_phys']
    Bz_phys = tmp['Bz_phys']
    Pg_phys = tmp['Pgas_phys']

    Jx = tmp['Jx']
    Jy = tmp['Jy']
    Jz = tmp['Jz']
    Bx = tmp['Bx']
    By = tmp['By']
    Bz = tmp['Bz']
    Pg = tmp['Pgas']

    x = np.linspace(0, 62*0.16, 62)
    y = np.linspace(0, 62*0.16, 62)

    dx = 0.16 * 725.0
    dy = 0.16 * 725.0
    pixel_length = 20
    
    X, Y = np.meshgrid(x, y)

    U_phys = Jy_phys * Bz_phys - Jz_phys * By_phys
    V_phys = Jz_phys * Bx_phys - Jx_phys * Bz_phys

    dPg_dx_phys = (Pg_phys[2:, 1:-1] - Pg_phys[:-2, 1:-1]) / (2 * dx)
    dPg_dy_phys = (Pg_phys[1:-1, 2:] - Pg_phys[1:-1, :-2]) / (2 * dy)

    U = np.median(Jy * Bz - Jz * By, axis=0)
    V = np.median(Jz * Bx - Jx * Bz, axis=0)

    dPg_dx = np.median((Pg[:, 2:, 1:-1] - Pg[:, :-2, 1:-1]) / (2 * dx), axis=0)
    dPg_dy = np.median((Pg[:, 1:-1, 2:] - Pg[:, 1:-1, :-2]) / (2 * dy), axis=0)


    magnitudes = np.sqrt(U**2 + V**2)
    max_mag = np.max(magnitudes)    
    scaling_factor = max_mag / pixel_length

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True, layout='constrained')

    # Display the background
    img = ax[0, 0].imshow(np.log10(Pg_phys[1:-1, 1:-1]), 
                extent=[0, 62*0.16, 0, 62*0.16], 
                origin='lower', 
                cmap='inferno')

    
    ax[0, 0].quiver(X, Y, U_phys[1:-1, 1:-1], V_phys[1:-1, 1:-1], 
          color='white',          # Set arrow color to red
          width=2,              # Increase thickness (in points)
          headwidth=3,          # Adjust head size relative to width
          headlength=4,         # Adjust head length relative to width
          units='dots', 
          scale=scaling_factor,
          scale_units='dots',
          pivot='mid')

    img = ax[0, 1].imshow(np.log10(np.median(Pg, axis=0))[1:-1, 1:-1], 
                extent=[0, 62*0.16, 0, 62*0.16], 
                origin='lower', 
                cmap='inferno')


    ax[0, 1].quiver(X, Y, U[1:-1, 1:-1], V[1:-1, 1:-1], 
          color='white',          # Set arrow color to red
          width=2,              # Increase thickness (in points)
          headwidth=3,          # Adjust head size relative to width
          headlength=4,         # Adjust head length relative to width
          units='dots', 
          scale=scaling_factor,
          scale_units='dots',
          pivot='mid')

    magnitudes = np.sqrt(dPg_dx_phys**2 + dPg_dy_phys**2)
    max_mag = np.max(magnitudes)    
    scaling_factor = max_mag / pixel_length
    
    img = ax[1, 0].imshow(np.log10(Pg_phys[1:-1, 1:-1]), 
                extent=[0, 62*0.16, 0, 62*0.16], 
                origin='lower', 
                cmap='inferno')
    ax[1, 0].quiver(X, Y, dPg_dx_phys, dPg_dy_phys, 
          color='white',          # Set arrow color to red
          width=2,              # Increase thickness (in points)
          headwidth=3,          # Adjust head size relative to width
          headlength=4,         # Adjust head length relative to width
          units='dots', 
          scale=scaling_factor,
          scale_units='dots',
          pivot='mid')

    img = ax[1, 1].imshow(np.log10(np.median(Pg, axis=0))[1:-1, 1:-1],
                extent=[0, 62*0.16, 0, 62*0.16], 
                origin='lower', 
                cmap='inferno')
    ax[1, 1].quiver(X, Y, dPg_dx, dPg_dy, 
          color='white',          # Set arrow color to red
          width=2,              # Increase thickness (in points)
          headwidth=3,          # Adjust head size relative to width
          headlength=4,         # Adjust head length relative to width
          units='dots', 
          scale=scaling_factor,
          scale_units='dots',
          pivot='mid')
    

    ax[0, 0].set_title('Sim', fontsize=16)
    ax[0, 1].set_title('IQUV', fontsize=16)
    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)

    cbar = fig.colorbar(img, ax=ax, location='right', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r'log P$_g$ [dex]', fontsize=14)

    if save:
        pl.savefig(f'figs/validation_lorentz.pdf', dpi=300)
    

if __name__ == "__main__":

    save = True

    # Horizontal maps
    plot_horizontal_validation(save=save, quantity=0)
    plot_horizontal_validation(save=save, quantity=5)
    plot_horizontal_validation(save=save, quantity=6)

    # Samples
    plot_horizontal_samples(save=save, quantity=1)
    
    # Vertical curves
    plot_vertical_validation(save=save)

    # Compute divergence for disambiguation and compute dissipation
    # calculate_div()
    # calculate_dissipation()

    # Plot divergence and currents
    plot_div(save=save)
    
    # Plot Lorentz force and pressure gradient
    plot_lorentz(save=save)