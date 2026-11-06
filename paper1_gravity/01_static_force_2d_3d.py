#(Generates force_plot_2d_3d.png)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# =====================================================================
# Paper I - Figure 1: Static Force Test (2D vs 3D Kernel)
# Description: Measures the gravitational force between mass-vortices 
#              using a 2D Poisson kernel vs a 3D Green's kernel.
# =====================================================================

# --- Grid Setup ---
N = 512
L = 200.0
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# --- Padding Factor (eliminates periodic boundary effects) ---
pad = 2
Npad = N * pad
start = (Npad - N) // 2

# --- Vortex Creation (Sech Profile for Localization) ---
width = 1.5
amplitude = 1.5
r = np.sqrt(X**2 + Y**2)
density = (amplitude / np.cosh(r/width))**2

density_pad = np.zeros((Npad, Npad))
density_pad[start:start+N, start:start+N] = density

# --- Fourier Space Setup ---
dx_pad = L / Npad
kx = 2*np.pi*fftfreq(Npad, d=dx_pad)
ky = 2*np.pi*fftfreq(Npad, d=dx_pad)
KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)
K[0,0] = 1e-10

G = 1.0
rho_k = fft2(density_pad)

# --- 2D Poisson Kernel ---
phi_k_2d = -4*np.pi*G * rho_k / (K**2)
phi_k_2d[0,0] = 0
phi_2d = np.real(ifft2(phi_k_2d))[start:start+N, start:start+N]

# --- 3D Green's Kernel ---
phi_k_3d = -2*np.pi*G * rho_k / K
phi_k_3d[0,0] = 0
phi_3d = np.real(ifft2(phi_k_3d))[start:start+N, start:start+N]

# --- Force Measurement ---
positions = [5, 8, 10, 15, 20, 30, 40, 60]
forces_2d, forces_3d = [], []

print(f"{'r':<6} | {'F (2D)':<12} | {'F (3D)':<12} | {'r*F (2D)':<12} | {'r^2*F (3D)':<12}")
print("-" * 70)

for pos in positions:
    ix = int((pos + L/2) / dx)
    iy = N // 2
    F_2d = -(phi_2d[iy, ix+1] - phi_2d[iy, ix-1]) / (2*dx)
    F_3d = -(phi_3d[iy, ix+1] - phi_3d[iy, ix-1]) / (2*dx)
    forces_2d.append(abs(F_2d))
    forces_3d.append(abs(F_3d))
    print(f"{pos:<6} | {abs(F_2d):<12.5f} | {abs(F_3d):<12.5f} | {pos*abs(F_2d):<12.5f} | {pos**2*abs(F_3d):<12.5f}")

# --- Force Law Fit ---
pos_arr = np.array(positions, dtype=float)
f2d_arr, f3d_arr = np.array(forces_2d), np.array(forces_3d)
mask = pos_arr >= 8

n2d, logA2d = np.polyfit(np.log(pos_arr[mask]), np.log(f2d_arr[mask]), 1)
n3d, logA3d = np.polyfit(np.log(pos_arr[mask]), np.log(f3d_arr[mask]), 1)

print(f"\n2D Kernel Fit: n = {n2d:.4f} | Expected: -1.0")
print(f"3D Kernel Fit: n = {n3d:.4f} | Expected: -2.0")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.loglog(pos_arr, f2d_arr, 'bs', markersize=8, label=rf'2D Poisson Data (\(n={n2d:.2f}\))')
ax.loglog(pos_arr, f3d_arr, 'ro', markersize=8, label=rf'3D Green Data (\(n={n3d:.2f}\))')

r_fit = np.linspace(min(pos_arr), max(pos_arr), 100)
ax.loglog(r_fit, np.exp(logA2d) * r_fit**n2d, 'b--', alpha=0.6)
ax.loglog(r_fit, np.exp(logA3d) * r_fit**n3d, 'r--', alpha=0.6)

ax.set_xlabel(r'Separation Distance \(r\) (log scale)', fontsize=13)
ax.set_ylabel(r'Force \(F\) (log scale)', fontsize=13)
ax.set_title(r'Static Force Test: 2D vs 3D Kernel Response', fontsize=15)
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('force_plot_2d_3d.png', dpi=300)
print("\nPlot saved as: force_plot_2d_3d.png")
plt.show()