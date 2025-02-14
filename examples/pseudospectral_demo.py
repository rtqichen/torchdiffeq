import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Define a simple harmonic oscillator
class HarmonicOscillator:
    def __init__(self, k=1.0):
        self.k = k
        
    def __call__(self, t, y):
        # y[0] is position, y[1] is velocity
        return torch.stack([
            y[1],                  # dx/dt = v
            -self.k * y[0]        # dv/dt = -kx
        ])

# Initial conditions: [x0, v0]
y0 = torch.tensor([1., 0.])

# Time points (using more points for better resolution)
t = torch.linspace(0, 10, 1000)

# Solve using RPM method
with torch.no_grad():
    solution = odeint(HarmonicOscillator(), y0, t, method='rpm',
                     options={'n_points': 16})  # Using 16 Chebyshev points

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, solution[:, 0].numpy(), label='Position')
plt.plot(t, solution[:, 1].numpy(), label='Velocity')
plt.grid(True)
plt.legend()
plt.title('Harmonic Oscillator - Pseudospectral Solution')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Print maximum error compared to analytical solution
exact_x = torch.cos(t)
exact_v = -torch.sin(t)
max_error = torch.max(torch.abs(solution[:, 0] - exact_x))
print("Maximum error in position: {:.2e}".format(max_error))
