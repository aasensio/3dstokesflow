import numpy as np
import witt


import numpy as np

def compute_heating_rates(Jx, Jy, Jz, Bx, By, Bz, T, ne, ni, nn):
    """
    Computes Ohmic and Ambipolar heating rates in Gaussian cgs units.
    
    Inputs (can be scalars, 1D arrays, or 2D/3D data cubes):
    - Jx, Jy, Jz : Current density components (statA/cm^2)
    - Bx, By, Bz : Magnetic field components (Gauss)
    - T          : Temperature (K)
    - ne         : Electron number density (cm^-3)
    - ni         : Ion number density (cm^-3)
    - nn         : Neutral number density (cm^-3)
    
    Returns:
    - Q_ohm      : Ohmic heating rate (erg cm^-3 s^-1)
    - Q_amb      : Ambipolar heating rate (erg cm^-3 s^-1)
    """
    
    # ---------------------------------------------------------
    # 1. Fundamental Constants (Gaussian cgs)
    # ---------------------------------------------------------
    kB = 1.380649e-16      # Boltzmann constant (erg/K)
    me = 9.109383e-28      # Electron mass (g)
    e_charge = 4.8032e-10  # Elementary charge (statC)
    c = 2.99792458e10      # Speed of light (cm/s)
    mH = 1.6726219e-24     # Hydrogen mass (g)

    mn = mH
    mi = mH

    m_in = (mi * mn) / (mi + mn)  # Reduced mass for ion-neutral collisions
    m_en = (me * mn) / (me + mn)  # Reduced mass for electron-neutral collisions
        
    # Cross-sections and Coulomb Logarithm
    Sigma_en = 1.0e-15     # Electron-neutral cross-section (cm^2)
    Sigma_in = 5.0e-15     # Ion-neutral cross-section (cm^2)

    T_ev = 8.617e-5 * T
    ln_Lambda = 23.4 - 1.15 * np.log10(ne) + 3.45 * np.log10(T_ev) # Coulomb logarithm (approximate)
    # ln_Lambda = 10.0       # Coulomb logarithm (approximate)
    
    # ---------------------------------------------------------
    # 2. Derived Thermodynamic Quantities
    # ---------------------------------------------------------
    # Mass densities
    rho_i = ni * mH
    rho_n = nn * mH
    rho_e = ne * me
    rho = rho_i + rho_n
    
    # Neutral fraction (handle division by zero if plasma is empty)
    xi_n = np.zeros_like(rho)
    mask_rho = rho > 0
    xi_n[mask_rho] = rho_n[mask_rho] / rho[mask_rho]
    
    # ---------------------------------------------------------
    # 3. Collision Frequencies (s^-1)
    # ---------------------------------------------------------
    # Electron-Ion (nu_ei)
    term1_ei = (4.0 / 3.0) * np.sqrt(2.0 * np.pi / me)
    term2_ei = (ni * e_charge**4 * ln_Lambda)
    term3_ei = (kB * T)**1.5
    nu_ei = term1_ei * term2_ei / term3_ei
        
    # Electron-Neutral (nu_en)
    nu_en = nn * Sigma_en * np.sqrt((8.0 * kB * T) / (np.pi * m_en))
    
    # Ion-Neutral (nu_in)
    nu_in = nn * Sigma_in * np.sqrt((8.0 * kB * T) / (np.pi * m_in))
    
    # ---------------------------------------------------------
    # 4. Ohmic Heating (Q_ohm)
    # ---------------------------------------------------------
    # Ohmic resistivity (seconds) - Avoid division by zero where ne=0
    eta = np.zeros_like(ne)
    mask_ne = ne > 0
    eta[mask_ne] = (me * (nu_ei[mask_ne] + nu_en[mask_ne])) / (ne[mask_ne] * e_charge**2)
    
    # Current magnitude squared
    J_sq = Jx**2 + Jy**2 + Jz**2
    
    # Ohmic dissipation (erg cm^-3 s^-1)
    Q_ohm = eta * J_sq
    
    # ---------------------------------------------------------
    # 5. Ambipolar Heating (Q_amb)
    # ---------------------------------------------------------
    # Lorentz Force components (dynes/cm^3) -- Note the 1/c factor
    FL_x = (Jy * Bz - Jz * By) / c
    FL_y = (Jz * Bx - Jx * Bz) / c
    FL_z = (Jx * By - Jy * Bx) / c
    
    # Lorentz force magnitude squared
    FL_sq = FL_x**2 + FL_y**2 + FL_z**2
    
    # Ambipolar dissipation (erg cm^-3 s^-1)
    # Avoid division by zero in fully ionized regions (where nu_in or rho_i = 0)
    Q_amb = np.zeros_like(rho)
    denominator = rho_i * nu_in + rho_e * nu_en
    mask_amb = denominator > 0
    
    Q_amb[mask_amb] = (xi_n[mask_amb]**2 / denominator[mask_amb]) * FL_sq[mask_amb]
    
    return Q_ohm, Q_amb

if __name__ == "__main__":
    # Example usage with test values
    Jx = np.array([1e-3])  # statA/cm^2
    Jy = np.array([1e-3])  # statA/cm^2
    Jz = np.array([1e-3])  # statA/cm^2
    Bx = np.array([100])   # Gauss
    By = np.array([100])   # Gauss
    Bz = np.array([100])   # Gauss
    T = np.array([5000])   # K
    ne = np.array([1e10])  # cm^-3
    ni = np.array([1e10])  # cm^-3
    nn = np.array([1e12])  # cm^-3
    
    Q_ohm, Q_amb = compute_heating_rates(Jx, Jy, Jz, Bx, By, Bz, T, ne, ni, nn)
    
    print("Ohmic Heating Rate (erg cm^-3 s^-1):", Q_ohm)
    print("Ambipolar Heating Rate (erg cm^-3 s^-1):", Q_amb)

# eos = witt.witt()

# temp = 5000.
# Pgas = 10000.

# Pe = eos.pe_from_pg(temp,Pgas)

# Na = (Pgas - Pe) / (eos.BK * temp)
# Ne = Pe          / (eos.BK * temp)