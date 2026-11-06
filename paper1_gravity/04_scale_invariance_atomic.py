#(Generates atomic_scale_plot.png)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# =====================================================================
# Paper I - Figure 4: Scale Invariance (Atomic Scale)
# Description: Runs the exact same model on a microscopic grid to prove
#              that the 1/r^2 law is scale-independent.
# =====================================================================

# --- Atomic Scale Grid Setup ---
N = 512
L = 5.0          # Very small domain
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# --- Padding Setup ---
pad = 2
Npad = N * pad
start = (Npad - N) // 2

# --- Atomic Scale Vortex ---
width = 0.05     # Extremely narrow
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
phi_k_3d = -2*np.pi*G * rho_k / K
phi_k_3d[0,0] = 0
phi_3d = np.real(ifft2(phi_k_3d))[start:start+N, start:start+N]

# --- Force Measurement ---
positions = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
forces = []

print(f"{'Distance (r)':<15} | {'Force (F)':<15} | {'r^2 * F':<15}")
print("-" * 50)

for pos in positions:
    ix = int((pos + L/2) / dx)
    iy = N // 2
    F_3d = -(phi_3d[iy, ix+1] - phi_3d[iy, ix-1]) / (2*dx)
    forces.append(abs(F_3d))
    print(f"{pos:<15} | {abs(F_3d):<15.4f} | {pos**2*abs(F_3d):<15.4f}")

# --- Fit ---
pos_arr = np.array(positions, dtype=float)
f_arr = np.array(forces)
mask = pos_arr >= 0.3
n, logA = np.polyfit(np.log(pos_arr[mask]), np.log(f_arr[mask]), 1)

print(f"\nAtomic Scale Fit: n = {n:.4f} | Expected: -2.0")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(pos_arr, f_arr, 'bo', markersize=8, label='Atomic Scale Data')
r_fit = np.linspace(min(pos_arr), max(pos_arr), 100)
ax.loglog(r_fit, np.exp(logA) * r_fit**n, 'k--', label=f'Fit: n={n:.3f}')

ax.set_xlabel(r'Separation Distance \(r\) (log scale)', fontsize=12)
ax.set_ylabel(r'Force \(F\) (log scale)', fontsize=12)
ax.set_title(r'Scale Invariance Test: Atomic/Sub-Atomic Scale', fontsize=14)
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('atomic_scale_plot.png', dpi=300)
print("Plot saved as: atomic_scale_plot.png")
plt.show()