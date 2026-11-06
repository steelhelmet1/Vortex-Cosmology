#(Generates Gravitational_waves.png)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# =====================================================================
# Paper I - Figure 3: Gravitational Wave Analogue
# Description: Simulates a binary system of orbiting vortices and records
#              energy density oscillations at a fixed sensor.
# =====================================================================

# --- Grid Setup ---
N = 256
L = 80.0
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

dt = 0.001
steps = 3000

# --- Create Two Same-Charge Vortices ---
def vortex(x0, y0, charge=1):
    r = np.sqrt((X-x0)**2 + (Y-y0)**2)
    theta = np.arctan2(Y-y0, X-x0)
    return np.tanh(r/1.5) * np.exp(1j * charge * theta)

R = 5.0
psi = vortex(-R/2, 0, charge=1) + vortex(R/2, 0, charge=1)

# --- Fourier Space Setup ---
kx = 2*np.pi*fftfreq(N, d=dx)
ky = 2*np.pi*fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

def kinetic_step(psi, dt):
    psi_k = fft2(psi)
    psi_k *= np.exp(-1j * K2 * dt)
    return ifft2(psi_k)

def potential_step(psi, dt):
    density = np.abs(psi)**2
    V = -density 
    return psi * np.exp(1j * V * dt)

# --- Sensor Setup ---
sensor_ix = int((30 + L/2) / dx)
sensor_iy = N // 2

time_steps = []
sensor_density = []

for step in range(steps):
    psi = kinetic_step(psi, dt/2)
    psi = potential_step(psi, dt)
    psi = kinetic_step(psi, dt/2)

    if step % 10 == 0:
        density = np.abs(psi)**2
        time_steps.append(step * dt)
        sensor_density.append(density[sensor_iy, sensor_ix])

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.plot(time_steps, sensor_density, 'b-', linewidth=1.5)
plt.title('Gravitational Wave Signal: Energy Density at Sensor', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Energy Density Oscillation', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Gravitational_waves.png', dpi=300)
print("Plot saved as: Gravitational_waves.png")
plt.show()