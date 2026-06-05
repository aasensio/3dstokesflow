import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py

def compute_J(Bx, By, Bz, dx, dy, dz):
        """
        Compute the current density J from the magnetic field components using finite differences.
        J = (1/μ₀) * (∇ × B)
        The components of J are given at two heights separated by dz
        Compute it at height 0
        """
        
        # Vertical derivatives
        dBx_dz = (Bx[:, 0, :, :] - Bx[:, 1, :, :]) / dz
        dBy_dz = (By[:, 0, :, :] - By[:, 1, :, :]) / dz
        dBz_dz = (Bz[:, 0, :, :] - Bz[:, 1, :, :]) / dz

        Bx_pad = F.pad(Bx, (1, 1, 1, 1))
        By_pad = F.pad(By, (1, 1, 1, 1))
        Bz_pad = F.pad(Bz, (1, 1, 1, 1))

        # Horizontal derivatives
        dBx_dy = (Bx_pad[:, 0, 1:-1, 2:] - Bx_pad[:, 0, 1:-1, :-2]) / (2 * dx)
        dBx_dx = (Bx_pad[:, 0, 2:, 1:-1] - Bx_pad[:, 0, :-2, 1:-1]) / (2 * dy)
        dBy_dy = (By_pad[:, 0, 1:-1, 2:] - By_pad[:, 0, 1:-1, :-2]) / (2 * dx)
        dBy_dx = (By_pad[:, 0, 2:, 1:-1] - By_pad[:, 0, :-2, 1:-1]) / (2 * dy)
        dBz_dx = (Bz_pad[:, 0, 2:, 1:-1] - Bz_pad[:, 0, :-2, 1:-1]) / (2 * dy)
        dBz_dy = (Bz_pad[:, 0, 1:-1, 2:] - Bz_pad[:, 0, 1:-1, :-2]) / (2 * dx)

        # Conversion factor for J calculation (assuming B in Gauss and distances in km)
        factor_J = (1e-4 / 1e3) / (4.0 * np.pi * 1e-7)  # Convert to cgs units for J calculation
        
        Jx = factor_J * (dBz_dy - dBy_dz)
        Jy = factor_J * (dBx_dz - dBz_dx)
        Jz = factor_J * (dBy_dx - dBx_dy)

        return Jx, Jy, Jz


class DifferentiableME0(nn.Module):
    def __init__(self, Bx_obs, By_obs, Bz_obs, dx, dy, dz, device='cpu'):
        """
        Bx_obs, By_obs: 2D Tensors of unresolved transverse fields (e.g., absolute values).
        Bz_obs: 2D Tensor of the vertical component of B.
        dBz_dz: 2D Tensor of the vertical derivative of Bz.
        dx, dy: Grid spacing constants.
        """
        super().__init__()
        # Freeze the observational data so PyTorch doesn't try to change it
        self.Bx_obs = Bx_obs.detach()
        self.By_obs = By_obs.detach()
        self.Bz_obs = Bz_obs.detach()        
            
        self.dx = dx
        self.dy = dy
        self.dz = dz

        nb, nh, nx, ny = Bx_obs.shape

        if device == 'cuda':
            self.Bx_obs = self.Bx_obs.cuda()
            self.By_obs = self.By_obs.cuda()
            self.Bz_obs = self.Bz_obs.cuda()
            self.theta = nn.Parameter(torch.randn((nb, 1, nx, ny)).cuda())
        else:
            self.theta = nn.Parameter(torch.randn((nb, 1, nx, ny)))

        
        # This is our trainable parameter. We initialize it randomly.
        # It represents the continuous "state" of the ambiguity resolution.

        # Conversion factor for J calculation (assuming B in Gauss and distances in km)
        self.factor_J = (1e-4 / 1e3) / (4.0 * np.pi * 1e-7)  # Convert to cgs units for J calculation
        
        
    def compute_spatial_derivatives(self, Bx, By, Bz):
        """
        Compute horizontal derivatives using central finite differences.
        PyTorch's F.pad and torch.roll act as a vectorized way to do this.
        """
        # Pad boundaries to handle edges (simplified as zero padding here)                
        Bx_pad = F.pad(Bx, (1, 1, 1, 1))
        By_pad = F.pad(By, (1, 1, 1, 1))
        Bz_pad = F.pad(Bz, (1, 1, 1, 1))

        # Central difference: f(x+1) - f(x-1) / 2*dx
        # Choose height 0
        dBx_dy = (Bx_pad[:, 0, 1:-1, 2:] - Bx_pad[:, 0, 1:-1, :-2]) / (2 * self.dx)
        dBx_dx = (Bx_pad[:, 0, 2:, 1:-1] - Bx_pad[:, 0, :-2, 1:-1]) / (2 * self.dy)
        dBy_dy = (By_pad[:, 0, 1:-1, 2:] - By_pad[:, 0, 1:-1, :-2]) / (2 * self.dx)
        dBy_dx = (By_pad[:, 0, 2:, 1:-1] - By_pad[:, 0, :-2, 1:-1]) / (2 * self.dy)
        dBz_dx = (Bz_pad[:, 0, 2:, 1:-1] - Bz_pad[:, 0, :-2, 1:-1]) / (2 * self.dy)
        dBz_dy = (Bz_pad[:, 0, 1:-1, 2:] - Bz_pad[:, 0, 1:-1, :-2]) / (2 * self.dx)

        dBx_dz = (Bx[:, 1, :, :] - Bx[:, 0, :, :]) / self.dz
        dBy_dz = (By[:, 1, :, :] - By[:, 0, :, :]) / self.dz
        dBz_dz = (Bz[:, 1, :, :] - Bz[:, 0, :, :]) / self.dz
                
        return dBx_dx, dBx_dy, dBy_dx, dBy_dy, dBz_dx, dBz_dy, dBx_dz, dBy_dz, dBz_dz

    def forward(self, T, border):
        # 1. Map continuous theta to a range of [-1, 1]
        S = torch.tanh(self.theta / T)
        
        # 2. Apply the sign modifier to the observed fields
        Bx = S * self.Bx_obs
        By = S * self.By_obs
        Bz = self.Bz_obs  # Bz is not modified by the sign ambiguity in this setup
        
        # 3. Compute gradients
        dBx_dx, dBx_dy, dBy_dx, dBy_dy, dBz_dx, dBz_dy, dBx_dz, dBy_dz, dBz_dz = self.compute_spatial_derivatives(Bx, By, Bz)
        
        # 4. Compute Jz and Divergence
        # Jz ~ dBy/dx - dBx/dy
        Jx = dBz_dy - dBy_dz
        Jy = dBx_dz - dBz_dx
        Jz = dBy_dx - dBx_dy

        # Div(B) = dBx/dx + dBy/dy + dBz/dz
        divB = dBx_dx + dBy_dy + dBz_dz
        
        # 5. Calculate total energy (Loss)
        energy_J = torch.mean(Jz[:, border:-border, border:-border]**2)  # Ignore boundaries for Jz
        energy_div = torch.mean(divB[:, border:-border, border:-border]**2)  # Ignore boundaries for divergence        

        J_abs = torch.sqrt(Jx[:, border:-border, border:-border]**2 + Jy[:, border:-border, border:-border]**2 + Jz[:, border:-border, border:-border]**2)

        divB_abs = torch.abs(divB[:, border:-border, border:-border])

        Jx = self.factor_J * Jx.detach()
        Jy = self.factor_J * Jy.detach()
        Jz = self.factor_J * Jz.detach()
        
        total_energy = torch.mean((J_abs + divB_abs)**2)
        return total_energy, energy_J, energy_div, S, divB, (Jx, Jy, Jz)
    
    def optimize(self, epochs=500, border=1, lr=0.01):

        # Annealing parameters
        T_initial = 10.0
        T_min = 0.05
        epochs_anneal = epochs / 2.0
        
        # Calculate decay factor so T reaches T_min exactly at the final epoch
        decay_rate = (T_min / T_initial) ** (1.0 / epochs_anneal)
        T = T_initial

        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Optimization loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, energy_J, energy_div, S_continuous, divB, (Jx, Jy, Jz) = self.forward(T, border)
            loss.backward()
            optimizer.step()

            # Cool down the temperature
            T = max(T_min, T * decay_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Energy: {loss.item():.4f} | J Energy: {energy_J.item()**2:.4f} | Div Energy: {energy_div.item()**2:.4f} | T: {T:.4f}")

        final_signs = torch.sign(S_continuous).detach()
        resolved_Bx = final_signs * self.Bx_obs
        resolved_By = final_signs * self.By_obs

        return resolved_Bx[:, 0, :, :], resolved_By[:, 0, :, :], final_signs[:, 0, :, :], divB.detach(), (Jx.detach(), Jy.detach(), Jz.detach())

        
if __name__ == "__main__":
    f = h5py.File('inversion/validation_results.h5', 'r')
    sol_stokes = f['sol'][:]
    phys = f['phys'][:]
    phys = np.transpose(phys, (2, 3, 0, 1))  # [B, H, X, Y]
    f.close()

    Bp1 = sol_stokes[:, 3, :, :]
    Bp2 = sol_stokes[:, 4, :, :]
    Bz = sol_stokes[:, 5, :, :]

    BT = np.sqrt(Bp1**2 + Bp2**2)
    twophi = np.atan2(Bp2, Bp1)
    phi = (0.5 * twophi) % np.pi
    Bx = BT * np.cos(phi)
    By = BT * np.sin(phi)

    Bp1 = phys[3, :, :, :]
    Bp2 = phys[4, :, :, :]
    Bz = phys[5, :, :, :]

    BT = np.sqrt(Bp1**2 + Bp2**2)
    twophi = np.atan2(Bp2, Bp1)
    phi = (0.5 * twophi) % np.pi
    Bx_phys = BT * np.cos(phi)
    By_phys = BT * np.sin(phi)

    model = DifferentiableME0(Bx, By, Bz, dx=1.0, dy=1.0, dz=1.0)
    Bx, By, signs, divB, J = model.optimize(epochs=500)