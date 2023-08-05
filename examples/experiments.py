import itertools
import os.path
import subprocess

# List of different arguments to try
networks = ['resnet', 'odenet']
batch_sizes = [128]
data_percentages = [0.25, 0.5, 0.75, 1.0]  # List of data percentages


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

venv_path = os.path.join(parent_dir, 'myenv', 'bin', 'activate')

for network, batch_size, data_percentage in itertools.product(networks, batch_sizes, data_percentages):
    adjoints = [False, True] if network == 'odenet' else [False]
    for adjoint in adjoints:
        cmd = [
            'source',  # Use 'source' to run a shell command in the current shell environment
            venv_path,
            '&&',  # To chain multiple commands
            'python',  # Replace with your main script filename
            'odenet_mnist.py',
            '--network', network,
            '--adjoint', str(adjoint),
            '--batch_size', str(batch_size),
            '--data_percentage', str(data_percentage)
        ]

        # Run the command using subprocess
        subprocess.run(' '.join(cmd), shell=True)