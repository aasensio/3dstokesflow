from curses import panel
from turtle import up

import torch
import numpy as np
import matplotlib.pyplot as pl
import h5py
import interpolation
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.lines import Line2D
import disambig
from einops import rearrange
import av
import cv2
import torch.nn.functional as F
import witt
from tqdm import tqdm
import dissipation


def mad(x, axis=0):
    median = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - median), axis=axis)
    return mad

def flatten_hinode(x, nx, ny, ltau):
    ltau = np.array(ltau)
    ntau = len(ltau)
    labels = ['T', 'v', 'T', 'Bx', 'By', 'Bz']
    
    out = np.zeros((6, ntau, nx, ny))

    for j in range(6):
        for i in range(ntau):        
            ind = np.argmin(np.abs(ltau[i] - x['log_tau'][:]))
            out[j, i, :, :] = x[labels[j]][:, 0, ind].reshape((nx, ny))

    return out
        
def plot_z_hinode_qs(save=False, quantity=0):
    
    f = h5py.File('inversion/hinode_results_qs.h5', 'r')
    sol_stokes = f['sol'][:]
    f.close()
    
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5,  -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variable_names = ['T', 'v_z', 'h', 'B_x', 'B_y', 'B_z']
    variables = ['T', r'v$_z$', 'h', r'B$_x$', r'B$_y$', r'B$_z$'] 
    units = ['K', 'km/s', 'km', 'G', 'G', 'G']

    # Define the z axis
    z = np.array([100.0, 200.0, 300.0])
    z = torch.tensor(z, dtype=torch.float32)
    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...][:, ::-1, :, :].copy()
    T = sol_stokes[:, 0, ...]
    vz = sol_stokes[:, 1, ...]
    Bp1 = sol_stokes[:, 3, ...]
    Bp2 = sol_stokes[:, 4, ...]
    Bz = sol_stokes[:, 5, ...]
    z0_stokes = np.mean(h[:, 2, :, :])
    
    # Interpolation
    T = torch.tensor(T, dtype=torch.float32)
    vz = torch.tensor(vz, dtype=torch.float32)
    Bp1 = torch.tensor(Bp1, dtype=torch.float32)
    Bp2 = torch.tensor(Bp2, dtype=torch.float32)
    Bz = torch.tensor(Bz, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)
    T = interpolation.interpolate_heights(T, h - z0_stokes, z, fill_value=np.nan)
    vz = interpolation.interpolate_heights(vz, h - z0_stokes, z, fill_value=np.nan)
    Bp1 = interpolation.interpolate_heights(Bp1, h - z0_stokes, z, fill_value=np.nan)
    Bp2 = interpolation.interpolate_heights(Bp2, h - z0_stokes, z, fill_value=np.nan)
    Bz = interpolation.interpolate_heights(Bz, h - z0_stokes, z, fill_value=np.nan)

    B = torch.sqrt(Bp1**2 + Bp2**2 + Bz**2)
    theta_B = torch.acos(Bz / B) * 180 / np.pi

    vars = [T, vz, theta_B, Bz]
                
    nx, ny = sol_stokes.shape[3], sol_stokes.shape[4]
    fig, ax = pl.subplots(nrows=6, ncols=4, figsize=(17.0, 23), layout='constrained', sharex=True, sharey=True)    

    quantities = [0, 1, 3, 5]
    panelx = [0, 3, 0, 3]
    panely = [0, 0, 2, 2]

    for j in range(4):
        
        if j == 0:
            minval = [4000, 4000, 4000]
            maxval = [6500, 5500, 5500]
            minval_mad = [0, 0, 0]
            maxval_mad = [400, 400, 700]
            cmap = 'inferno'
            label_color = ['white', 'white']

        if j == 1:
            minval = [-5, -4, -3]
            maxval = [5, 4, 3]
            minval_mad = [0, 0, 0]
            maxval_mad = [2.0, 2.0, 2.0]
            cmap = 'RdBu_r'
            label_color = ['black', 'black']

        if j == 2:
            minval = [0, 0, 0]
            maxval = [180, 180, 180]
            minval_mad = [0, 0, 0]
            maxval_mad = [50, 50, 50]
            cmap = 'RdBu_r'
            label_color = ['black', 'black']

        if j == 3:
            minval = [-200, -200, -200]
            maxval = [200, 200, 200]
            minval_mad = [0, 0, 0]
            maxval_mad = [40, 40, 60]
            cmap = 'RdBu_r'
            label_color = ['black', 'black']

        low = 2
        up = -2
        nx, ny = vars[0][:, :, low:up, low:up].shape[2], vars[0][:, :, low:up, low:up].shape[3]
            
        for i in range(3):

            cmap_obj = pl.get_cmap(cmap).copy()
            cmap_obj.set_bad(color='black')  # masked values will be black
                                    
            tmp = np.nanmedian(vars[j][:, i, low:up, low:up], axis=0)
            tmp = np.ma.masked_where(tmp == np.nan, tmp)
            im = ax[i+panelx[j], 0+panely[j]].imshow(tmp, vmin=minval[i], vmax=maxval[i], extent=[0, nx*0.16, 0, ny*0.16], cmap=cmap_obj, origin='lower')    
            cbar = fig.colorbar(im, ax=ax[i+panelx[j], 0+panely[j]], location='right', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12)        

            cmap_obj = pl.get_cmap('inferno').copy()
            cmap_obj.set_bad(color='black')  # masked values will be black

            tmp = mad(vars[j][:, i, low:up, low:up], axis=0)
            tmp = np.ma.masked_where(tmp == np.nan, tmp)
            im = ax[i+panelx[j], 1+panely[j]].imshow(tmp, vmin=minval_mad[i], vmax=maxval_mad[i], extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
            cbar = fig.colorbar(im, ax=ax[i+panelx[j], 1+panely[j]], location='right', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12)        
            
            # im = ax[i+panelx[j], 2+panely[j]].imshow(tmp_sir, vmin=minval[i], vmax=maxval[i], extent=[0, nx*0.16, 0, nx*0.16], cmap=cmap)    
            # cbar = fig.colorbar(im, ax=ax[i+panelx[j], 2+panely[j]], location='right', fraction=0.046, pad=0.04)
            # cbar.ax.tick_params(labelsize=12)

    ax[0, 0].set_title('Median')
    ax[0, 1].set_title('MAD')
    # ax[0, 2].set_title('SIR')
    ax[3, 0].set_title('Median')
    ax[3, 1].set_title('MAD')
    # ax[3, 2].set_title('SIR')
    ax[0, 2].set_title('Median')
    ax[0, 3].set_title('MAD')
    # ax[0, 5].set_title('SIR')
    ax[3, 2].set_title('Median')
    ax[3, 3].set_title('MAD')
    # ax[3, 5].set_title('SIR')

    ax[0, 0].text(0.0, 1.10, 'T [K]',
                        transform=ax[0, 0].transAxes, 
                        fontsize=13, 
                        verticalalignment='top', 
                        color='black',
                        fontweight='bold')
    
    ax[0, 2].text(0.0, 1.10, r'$\theta_B$ [deg]',
                        transform=ax[0, 2].transAxes, 
                        fontsize=13, 
                        verticalalignment='top', 
                        color='black',
                        fontweight='bold')
    
    ax[3, 0].text(0.0, 1.10, 'v [km/s]',
                        transform=ax[3, 0].transAxes, 
                        fontsize=13, 
                        verticalalignment='top', 
                        color='black',
                        fontweight='bold')

    ax[3, 2].text(0.0, 1.10, r'$B_z$ [G]',
                        transform=ax[3, 2].transAxes, 
                        fontsize=13, 
                        verticalalignment='top', 
                        color='black',
                        fontweight='bold')
    
    for i in range(3):
        ax[i, 0].text(0.05, 0.1, f'{z[i]:.0f} km',
                        transform=ax[i, 0].transAxes, 
                        fontsize=18, 
                        verticalalignment='top', 
                        color='white',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'))

    

    #     cbar = fig.colorbar(im, ax=ax[i, :], location='right', fraction=0.046, pad=0.04)
    #     cbar.ax.tick_params(labelsize=12)
    #     cbar.set_label(f'{variables[quantity]} [{units[quantity]}]', fontsize=14)
        
    #     ax[i, 0].text(0.05, 0.1, f'{z[i]:.0f} km',
    #                         transform=ax[i, 0].transAxes, 
    #                         fontsize=18, 
    #                         verticalalignment='top', 
    #                         color=label_color[0],
    #                         fontweight='bold')
            
    # for i in range(2):
    #     ax[0, i].text(0.05, 0.95, f'{labels[i]}',
    #                         transform=ax[0, i].transAxes, 
    #                         fontsize=18, 
    #                         verticalalignment='top', 
    #                         color=label_color[1],
    #                         fontweight='bold')
        
    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    # fig.canvas.draw()
    # x_left = min(a.get_position().x0 for a in ax.flat)
    # x_right = max(a.get_position().x1 for a in ax.flat)
    # y_bottom = min(a.get_position().y0 for a in ax.flat) - 0.01
    # y_top = max(a.get_position().y1 for a in ax.flat) + 0.01
    # x_sep = 0.5 * (ax[0, 2].get_position().x1 + ax[0, 3].get_position().x0) + 0.015
    # y_sep = 0.5 * (ax[2, 0].get_position().y0 + ax[3, 0].get_position().y1) + 0.0
    # fig.add_artist(Line2D([x_sep, x_sep], [y_bottom, y_top], transform=fig.transFigure, color='green', linewidth=3.5))
    # fig.add_artist(Line2D([x_left, x_right], [y_sep, y_sep], transform=fig.transFigure, color='green', linewidth=3.5))

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=14)

    if save:
        pl.savefig(f'figs/hinode_flow.pdf', dpi=300, bbox_inches='tight')
        

def calculate_div(save=False):
    f = h5py.File('inversion/hinode_results_qs.h5', 'r')
    sol_stokes = f['sol'][:]
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
    Bp1 = sol_stokes[:, 3, ...]
    Bp2 = sol_stokes[:, 4, ...]
    Bz = sol_stokes[:, 5, ...]
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


    model = disambig.DifferentiableME0(Bx_stokes, 
                                       By_stokes, 
                                       Bz_stokes,                                       
                                       dx=deltax, 
                                       dy=deltay,
                                       dz=deltaz, 
                                       device='cuda')
    
    Bx, By, signs, divB, (Jx, Jy, Jz) = model.optimize(epochs=5000, border=1, lr=0.01)

    np.savez('hinode_div.npz', divB=divB.cpu().numpy(), Jx=Jx.cpu().numpy(), Jy=Jy.cpu().numpy(), Jz=Jz.cpu().numpy(), Bx=Bx.cpu().numpy(), By=By.cpu().numpy())


def calculate_dissipation():

    f = h5py.File('inversion/hinode_results_qs.h5', 'r')
    sol_stokes = f['sol'][:]             
    f.close()
    
    # Define the z axis
    z = np.array([90.0, 100.0])
    z = torch.tensor(z, dtype=torch.float32)
    
    ##############
    # Neural inversion
    # conditioned on full Stokes
    ##############
    h = sol_stokes[:, 2, ...][:, ::-1, :, :].copy()
    T = sol_stokes[:, 0, ...].copy()
    Bz = sol_stokes[:, 5, ...].copy()
    logP = sol_stokes[:, 6, ...].copy()
    z0_stokes = np.mean(h[:, 2, :, :])

    T = torch.tensor(T, dtype=torch.float32)
    logP = torch.tensor(logP, dtype=torch.float32)
    Bz = torch.tensor(Bz, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)    
    T = interpolation.interpolate_heights(T, h - z0_stokes, z, fill_value=0.0)
    logP = interpolation.interpolate_heights(logP, h - z0_stokes, z, fill_value=0.0)
    Bz = interpolation.interpolate_heights(Bz, h - z0_stokes, z, fill_value=0.0)
    
    data = np.load('hinode_div.npz')
    divB = data['divB']
    Jx = data['Jx'] * 3e5
    Jy = data['Jy'] * 3e5
    Jz = data['Jz'] * 3e5
    Bx = data['Bx']
    By = data['By']    
    Bz = Bz[:, 0, ...].cpu().numpy()
        
    eos = witt.witt()
        
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
            for k in range(25):
                if temp[k, i, j] != 0:
                    Pe[k, i, j] = eos.pe_from_pg(temp[k, i, j], Pgas[k, i, j])

    np.savez('hinode_quantities_for_dissipation.npz',
             Pe=Pe,
             Pgas=Pgas,
             temp=temp,
             Jx=Jx,
             Jy=Jy,
             Jz=Jz,
             Bx=Bx,
             By=By,
             Bz=Bz)

def plot_div(save=False):

    f = h5py.File('inversion/hinode_results_qs.h5', 'r')
    sol_stokes = f['sol'][:]    
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
    Bp1 = sol_stokes[:, 3, ...]
    Bp2 = sol_stokes[:, 4, ...]
    Bz = sol_stokes[:, 5, ...]
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
    
    data = np.load('hinode_div.npz')
    divB = data['divB']
    Jx = data['Jx']
    Jy = data['Jy']
    Jz = data['Jz']
    Bx = data['Bx']
    By = data['By']

    low = 2
    up = -2
    nx, ny = Bp1_stokes[:, :, low:up, low:up].shape[2], Bp1_stokes[:, :, low:up, low:up].shape[3]

    
    fig, ax = pl.subplots(nrows=2, ncols=4, figsize=(25, 11), layout='constrained', sharex=True, sharey=True)
    im = ax[0, 0].imshow(np.median(Bp1_stokes[:, 0, low:up, low:up], axis=0), vmin=-150, vmax=150, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 0].imshow(mad(Bp1_stokes[:, 0, low:up, low:up], axis=0), vmin=0, vmax=40, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 0], location='right', fraction=0.046, pad=0.04)
    ax[0, 0].set_title(r'B$_{p1}$ [G]', fontsize=16)
        
    im = ax[0, 1].imshow(np.median(Bp2_stokes[:, 0, low:up, low:up], axis=0), vmin=-150, vmax=150, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(mad(Bp2_stokes[:, 0, low:up, low:up], axis=0), vmin=0, vmax=40, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 1], location='right', fraction=0.046, pad=0.04)
    ax[0, 1].set_title(r'B$_{p2}$ [G]', fontsize=16)
    
    im = ax[0, 2].imshow(np.median(Bx[:, low:up, low:up], axis=0), vmin=-150, vmax=150, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 2], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 2].imshow(mad(Bx[:, low:up, low:up], axis=0), vmin=0, vmax=60, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 2], location='right', fraction=0.046, pad=0.04)
    ax[0, 2].set_title(r'B$_x$ [G]', fontsize=16)
    
    
    im = ax[0, 3].imshow(np.median(By[:, low:up, low:up], axis=0), vmin=-150, vmax=150, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 3], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 3].imshow(mad(By[:, low:up, low:up], axis=0), vmin=0, vmax=60, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 3], location='right', fraction=0.046, pad=0.04)
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
    
    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)
    
    if save:
        pl.savefig(f'figs/hinode_Bxy.pdf', dpi=300)

    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(21, 12), layout='constrained', sharex=True, sharey=True)
    im = ax[0, 0].imshow(np.median(Jx[:, low:up, low:up], axis=0), vmin=-0.1, vmax=0.1, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(np.median(Jy[:, low:up, low:up], axis=0), vmin=-0.1, vmax=0.1, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[0, 2].imshow(np.median(Jz[:, low:up, low:up], axis=0), vmin=-0.05, vmax=0.05, extent=[0, ny*0.16, 0, nx*0.16], cmap='RdBu_r', origin='lower')
    pl.colorbar(im, ax=ax[0, 2], location='right', fraction=0.046, pad=0.04)
    ax[0, 0].set_title(r'J$_x$ [A/m$^2$]', fontsize=16)
    ax[0, 1].set_title(r'J$_y$ [A/m$^2$]', fontsize=16)
    ax[0, 2].set_title(r'J$_z$ [A/m$^2$]', fontsize=16)

    im = ax[1, 0].imshow(mad(Jx[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.05, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(mad(Jy[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.05, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 1], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 2].imshow(mad(Jz[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.02, extent=[0, ny*0.16, 0, nx*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 2], location='right', fraction=0.046, pad=0.04)

    
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
    
    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)

    if save:
        pl.savefig(f'figs/hinode_currents.pdf', dpi=300)


    ##########################################
    # DISSIPATION
    ##########################################
    data = np.load('hinode_quantities_for_dissipation.npz')    
    Pe = data['Pe']
    Pgas = data['Pgas']    
    temp = data['temp']    
    Jx = data['Jx']
    Jy = data['Jy']
    Jz = data['Jz']
    Bx = data['Bx']
    By = data['By']
    Bz = data['Bz']

    BK = 1.380649e-16  # Boltzmann constant in erg/K

    Na = (Pgas - Pe) / (BK * temp)
    Ne = Pe          / (BK * temp)
    Ni = Ne
        
    Q_ohm, Q_amb = dissipation.compute_heating_rates(Jx, Jy, Jz, Bx, By, Bz, temp, Ne, Ni, Na)

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(12, 10), layout='constrained', sharex=True, sharey=True)
    im = ax[0, 0].imshow(np.median(Q_ohm[:, low:up, low:up], axis=0), vmin=0., vmax=0.01, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[0, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(np.median(Q_amb[:, low:up, low:up], axis=0), vmin=0., vmax=0.0015, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[0, 1], location='right', fraction=0.046, pad=0.04)    
    ax[0, 0].set_title(r'Q$_{ohm}$ [erg cm$^{-3}$ s$^{-1}$]', fontsize=16)
    ax[0, 1].set_title(r'Q$_{amb}$ [erg cm$^{-3}$ s$^{-1}$]', fontsize=16)
    
    im = ax[1, 0].imshow(mad(Q_ohm[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.003, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 0], location='right', fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(mad(Q_amb[:, low:up, low:up], axis=0), vmin=0.0, vmax=0.0004, extent=[0, nx*0.16, 0, ny*0.16], cmap='inferno', origin='lower')
    pl.colorbar(im, ax=ax[1, 1], location='right', fraction=0.046, pad=0.04)

    
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
    
    fig.supxlabel('x [arcsec]', fontsize=16)
    fig.supylabel('y [arcsec]', fontsize=16)

    for axis in ax.flat:
        axis.tick_params(axis='both', which='major', labelsize=16)

    if save:
        pl.savefig(f'figs/hinode_dissipation.pdf', dpi=300)

def plot_loop_hinode(loop=1, cut_left=0, save=False):
    f = h5py.File(f'hinode_loop/loop{loop}.h5', 'r')
    sol_stokes = f['geom'][:]
    f.close()

    if loop == 1:
        start = 0
        cut_left = 3
    else:
        start = 3
        cut_left = 3
    
    tau = np.array([1. , 0.5,  0. ,  -0.5,  -1. ,  -1.5,  -2. ,  -2.5,  -3. ,  -3.5,  -4.])
    variables = ['T', 'vz', 'Bz', 'BT'] 
    z = [0, 100, 200, 300]
    
    ind_z = [2, 6, 10, 14]
    tmp = sol_stokes[2, 3, 0, 0, :, cut_left:]
    nx, ny = tmp.shape[0], tmp.shape[1]

    # Percentile, variable, time, z, x, y        
    fig, ax = pl.subplots(nrows=20, ncols=4, figsize=(13.0, 20), layout='constrained', sharex=True, sharey=False)
    for j in range(10):
        for i in range(4):
            Bt = np.sqrt(sol_stokes[2, 2, j+start, ind_z[i], :, cut_left:]**2 + sol_stokes[2, 3, j+start, ind_z[i], :, cut_left:]**2)
            ax[j, i].imshow(Bt, extent=[0, nx*0.16, 0, ny*0.16], cmap='viridis', origin='lower', vmin=0, vmax=50, aspect='auto')

    for j in range(10):
        for i in range(4):
            ax[j+10, i].imshow(sol_stokes[2, 4, j+start, ind_z[i], :, cut_left:], extent=[0, nx*0.16, 0, ny*0.16], cmap='RdBu_r', origin='lower', vmin=-70, vmax=70, aspect='auto')

    # fig.subplots_adjust(wspace=0.0, hspace=0.0)
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Add two colorbars to the right of all panels
    # First colorbar for the top half (rows 0-9, Bz)
    sm_top = pl.cm.ScalarMappable(cmap='viridis', norm=pl.Normalize(vmin=0, vmax=50))
    sm_top.set_array([])
    cbar_top = fig.colorbar(sm_top, ax=ax[:10, :], location='right', fraction=0.02, pad=0.02)
    cbar_top.set_label(r'B$_t$ [G]', fontsize=12)
    cbar_top.ax.tick_params(labelsize=10)

    # Second colorbar for the bottom half (rows 10-19, Bz)
    sm_bot = pl.cm.ScalarMappable(cmap='RdBu_r', norm=pl.Normalize(vmin=-70, vmax=70))
    sm_bot.set_array([])
    cbar_bot = fig.colorbar(sm_bot, ax=ax[10:, :], location='right', fraction=0.02, pad=0.02)
    cbar_bot.set_label(r'B$_z$ [G]', fontsize=12)
    cbar_bot.ax.tick_params(labelsize=10)

    for i in range(4):
        ax[0, i].text(
            0.05, 0.93, f'{z[i]:.0f} km',
            transform=ax[0, i].transAxes,            
            fontsize=18,
            verticalalignment='top',
            color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

    for i in range(10):
        ax[i, -1].text(
            0.95, 0.25, f't={30*i:.0f} s',
            transform=ax[i, -1].transAxes,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='right',
            color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none')
        )

    if save:
        pl.savefig(f'figs/hinode_loop{loop}_panels.pdf', dpi=300, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":

    pl.close('all')
    # calculate_div(save=False)
    # calculate_dissipation()
    # plot_div(save=True)
    plot_z_hinode_qs(save=True)

    # plot_loop_hinode(loop=1, save=True)