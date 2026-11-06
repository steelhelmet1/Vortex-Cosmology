#(Generates force_plot_2d_3d.png)


import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt

# =====================================================================
# Paper I - Figure 2: True 3D Simulation
# Description: Executes a full 3D simulation without 2D hacks to 
#              validate the inverse-square law in true 3D space.
# =====================================================================

# --- True 3D Grid Setup ---
N = 96
L = 40.0
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
z = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- Create a 3D Soliton ---
def create_soliton(x0, y0, z0, amplitude=1.5, width=1.5):
    r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
    return amplitude / np.cosh(r/width)

psi = create_soliton(0, 0, 0)
density = np.abs(psi)**2

# --- 3D Fourier Space Setup ---
kx = 2*np.pi*fftfreq(N, d=dx)
ky = 2*np.pi*fftfreq(N, d=dx)
kz = 2*np.pi*fftfreq(N, d=dx)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0,0,0] = 1e-10

# --- True 3D Poisson Equation ---
G = 1.0
rho_k = fftn(density)
phi_k = -4 * np.pi * G * rho_k / K2
phi_k[0,0,0] = 0
phi = np.real(ifftn(phi_k))

# --- Force vs Distance ---
distances = [3, 4, 5, 6, 8, 10, 12, 15]
forces = []
center_idx = N // 2

print(f"{'Distance (r)':<15} | {'Force (F)':<15} | {'r^2 * F':<15}")
print("-" * 50)

for d in distances:
    idx = center_idx + int(d / dx)
    Fx = -(phi[idx+1, center_idx, center_idx] - phi[idx-1, center_idx, center_idx]) / (2 * dx)
    force = abs(Fx)
    forces.append(force)
    print(f"{d:<15} | {force:<15.4f} | {d**2 * force:<15.4f}")

# --- Fit ---
pos_arr = np.array(distances, dtype=float)
force_arr = np.array(forces)
mask = pos_arr >= 4
n, logA = np.polyfit(np.log(pos_arr[mask]), np.log(force_arr[mask]), 1)

print(f"\nTrue 3D Gravity Fit: n = {n:.4f} | Expected: -2.0")

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(density[center_idx, :, :], extent=[-L/2, L/2, -L/2, L/2], cmap='inferno')
plt.title('3D Density Slice (z=0)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.loglog(pos_arr, force_arr, 'mo', markersize=8, label='True 3D Data')
r_fit = np.linspace(min(pos_arr), max(pos_arr), 100)
plt.loglog(r_fit, np.exp(logA) * r_fit**n, 'k--', label=f'Fit: n={n:.3f}')
plt.xlabel('Distance r (log scale)')
plt.ylabel('Force F (log scale)')
plt.title('True 3D Gravity (No Hacks)')
plt.legend()
plt.grid(True, which='both', ls='--')

plt.tight_layout()
plt.savefig('3d_gravity_plot.png', dpi=300)
print("Plot saved as: 3d_gravity_plot.png")
plt.show()