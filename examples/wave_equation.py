import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('Wave equation demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4'], default='dopri5')
parser.add_argument('--n_grid', type=int, default=128,
                    help='Number of spatial grid points')
parser.add_argument('--c', type=float, default=1.0,
                    help='Wave speed')
parser.add_argument('--t_end', type=float, default=4.0,
                    help='Simulation end time')
parser.add_argument('--n_time', type=int, default=200,
                    help='Number of time points to save')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--train', action='store_true',
                    help='Train a Neural ODE to learn wave dynamics from data')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


class WaveEquationODE(nn.Module):
    """Method-of-lines discretization of the 1D wave equation.

    Converts the PDE  u_tt = c^2 * u_xx  into a first-order ODE system:
        du/dt = v
        dv/dt = c^2 * Laplacian(u)

    where the spatial Laplacian is approximated via second-order finite differences
    with periodic boundary conditions.

    State tensor layout: [u (n_grid,), v (n_grid,)]  stacked as (2*n_grid,).
    """

    def __init__(self, n_grid, c=1.0, dx=None):
        super(WaveEquationODE, self).__init__()
        self.n_grid = n_grid
        self.c2 = c ** 2
        self.dx2 = (2.0 * np.pi / n_grid) ** 2 if dx is None else dx ** 2

    def forward(self, t, state):
        u = state[:self.n_grid]
        v = state[self.n_grid:]

        # Laplacian with periodic boundary conditions (second-order central differences)
        laplacian = (torch.roll(u, -1) - 2 * u + torch.roll(u, 1)) / self.dx2

        du_dt = v
        dv_dt = self.c2 * laplacian

        return torch.cat([du_dt, dv_dt])


def initial_condition(x, n_grid):
    """Two Gaussian pulses traveling in opposite directions."""
    L = 2.0 * np.pi
    u0 = torch.exp(-40.0 * (x - L / 3.0) ** 2) + 0.5 * torch.exp(-40.0 * (x - 2.0 * L / 3.0) ** 2)
    v0 = torch.zeros(n_grid, dtype=u0.dtype, device=u0.device)
    return torch.cat([u0, v0])


def generate_data(n_grid, c, t_end, n_time):
    """Solve the wave equation forward using torchdiffeq."""
    L = 2.0 * np.pi
    x = torch.linspace(0, L, n_grid + 1, dtype=torch.float64)[:-1]  # periodic grid
    dx = L / n_grid

    dynamics = WaveEquationODE(n_grid, c=c, dx=dx).double().to(device)
    state0 = initial_condition(x, n_grid).double().to(device)
    t = torch.linspace(0, t_end, n_time, dtype=torch.float64).to(device)

    with torch.no_grad():
        states = odeint(dynamics, state0, t, method=args.method, rtol=1e-8, atol=1e-10)

    return x, t, states, dynamics


# ---------------------------------------------------------------------------
# Neural ODE for learning wave dynamics
# ---------------------------------------------------------------------------

class NeuralWaveODE(nn.Module):
    """A neural network that learns the right-hand side of the wave ODE system."""

    def __init__(self, n_grid, hidden=256):
        super(NeuralWaveODE, self).__init__()
        self.n_grid = n_grid
        self.net = nn.Sequential(
            nn.Linear(2 * n_grid, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * n_grid),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        return self.net(state)


def get_batch(true_states, t, batch_time, batch_size):
    n_time = true_states.shape[0]
    s = torch.from_numpy(
        np.random.choice(np.arange(n_time - batch_time, dtype=np.int64),
                         batch_size, replace=False))
    batch_y0 = true_states[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([true_states[s + i] for i in range(batch_time)], dim=0)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def visualize_forward(x, t, states, n_grid):
    """Visualize the forward wave equation solution as a space-time heatmap."""
    import matplotlib.pyplot as plt
    u_all = states[:, :n_grid].cpu().numpy()
    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Space-time heatmap
    ax = axes[0]
    im = ax.pcolormesh(x_np, t_np, u_all, shading='auto', cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Wave equation: u(x, t)')
    fig.colorbar(im, ax=ax)

    # Snapshots at selected times
    ax = axes[1]
    n_snapshots = 5
    indices = np.linspace(0, len(t_np) - 1, n_snapshots, dtype=int)
    for idx in indices:
        ax.plot(x_np, u_all[idx], label='t={:.2f}'.format(t_np[idx]))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Snapshots')
    ax.legend(fontsize=8)

    # Energy conservation check
    ax = axes[2]
    dx = x_np[1] - x_np[0]
    v_all = states[:, n_grid:].cpu().numpy()
    # Kinetic + gradient energy (should be conserved)
    kinetic = 0.5 * np.sum(v_all ** 2, axis=1) * dx
    # Use forward differences consistent with the 3-point Laplacian stencil
    du = np.roll(u_all, -1, axis=1) - u_all
    gradient = 0.5 * args.c ** 2 * np.sum(du ** 2, axis=1) / dx
    total_energy = kinetic + gradient
    ax.plot(t_np, total_energy, 'b-', linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('Energy')
    ax.set_title('Total energy (should be constant)')
    ax.set_ylim([0, total_energy.max() * 1.5])

    fig.tight_layout()
    makedirs('png')
    plt.savefig('png/wave_equation_forward.png', dpi=150)
    plt.show(block=False)
    plt.pause(0.5)
    print("Saved forward solution to png/wave_equation_forward.png")


def visualize_training(true_y, pred_y, x, n_grid, itr):
    """Visualize Neural ODE training progress."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # True solution snapshot (last time step)
    u_true = true_y[-1, 0, :n_grid].cpu().numpy()
    u_pred = pred_y[-1, 0, :n_grid].cpu().detach().numpy()
    x_np = x.cpu().numpy()

    ax = axes[0]
    ax.plot(x_np, u_true, 'g-', linewidth=2, label='True')
    ax.plot(x_np, u_pred, 'b--', linewidth=2, label='Learned')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Iter {:04d}: Last time step'.format(itr))
    ax.legend()

    # Error
    ax = axes[1]
    ax.plot(x_np, np.abs(u_true - u_pred), 'r-', linewidth=1)
    ax.set_xlabel('x')
    ax.set_ylabel('|error|')
    ax.set_title('Pointwise absolute error')

    fig.tight_layout()
    makedirs('png')
    plt.savefig('png/wave_{:04d}.png'.format(itr), dpi=100)
    plt.draw()
    plt.pause(0.001)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Device:", device)
    print("Method:", args.method)
    print("Grid points:", args.n_grid)
    print("Wave speed:", args.c)

    # Generate ground-truth data
    print("\nSolving wave equation with method of lines...")
    start = time.time()
    x, t, true_states, true_dynamics = generate_data(
        args.n_grid, args.c, args.t_end, args.n_time)
    elapsed = time.time() - start
    print("Forward solve completed in {:.2f}s".format(elapsed))
    print("Solution shape: {} (n_time, 2*n_grid)".format(true_states.shape))

    # Energy conservation check
    n_grid = args.n_grid
    dx = (2 * np.pi / n_grid)
    u_all = true_states[:, :n_grid].cpu().numpy()
    v_all = true_states[:, n_grid:].cpu().numpy()
    kinetic = 0.5 * np.sum(v_all ** 2, axis=1) * dx
    # Use forward differences consistent with the 3-point Laplacian stencil
    du = np.roll(u_all, -1, axis=1) - u_all
    gradient = 0.5 * args.c ** 2 * np.sum(du ** 2, axis=1) / dx
    total_energy = kinetic + gradient
    energy_drift = np.abs(total_energy[-1] - total_energy[0]) / total_energy[0] * 100
    print("Energy drift: {:.4f}%".format(energy_drift))

    if args.viz and not args.train:
        visualize_forward(x, t, true_states, n_grid)

    if args.train:
        # ------- Train Neural ODE to learn wave dynamics from data -------
        # Convert to float32 for efficient neural network training
        true_states_f = true_states.float()
        t_f = t.float()

        print("\nTraining Neural ODE to learn wave dynamics...")
        func = NeuralWaveODE(n_grid).to(device)
        optimizer = optim.Adam(func.parameters(), lr=1e-3)

        if args.viz:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 4))
            plt.show(block=False)

        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = get_batch(
                true_states_f, t_f, args.batch_time, args.batch_size)
            pred_y = odeint(func, batch_y0, batch_t).to(device)
            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()

            if itr % args.test_freq == 0:
                with torch.no_grad():
                    test_y0 = true_states_f[0:1].to(device)
                    test_t = t_f.to(device)
                    pred_y_full = odeint(func, test_y0, test_t)
                    test_loss = torch.mean(torch.abs(pred_y_full[:, 0, :] - true_states_f))
                    print('Iter {:04d} | Train Loss {:.6f} | Test Loss {:.6f}'.format(
                        itr, loss.item(), test_loss.item()))

                    if args.viz:
                        # Reshape for visualization
                        vis_true = true_states_f.unsqueeze(1)  # (T, 1, 2*N)
                        vis_pred = pred_y_full  # (T, 1, 2*N)
                        visualize_training(vis_true, vis_pred, x.float(), n_grid, itr)

        print("\nTraining complete.")
        if args.viz:
            # Final comparison
            with torch.no_grad():
                pred_final = odeint(func, true_states_f[0:1].to(device), t_f.to(device))
            visualize_forward(x.float(), t_f, pred_final[:, 0, :], n_grid)
