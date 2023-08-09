import itertools
import os.path
import subprocess

# List of different arguments to try
networks = ['odenet', 'resnet']
batch_sizes = [128]
#data_percentages = [0.25, 0.50, 0.75, 1.0]  # List of data percentages
data_percentages = [0.01, 0.05, 0.10, 0.15, 0.20]

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

venv_path = os.path.join(parent_dir, 'myenv', 'bin', 'activate')

#script_filenames = ['odenet_mnist.py', 'odenet_cifar10.py']
script_filenames = ['odenet_mnist.py']

for script_filename in script_filenames:
    for network, batch_size, data_percentage in itertools.product(networks, batch_sizes, data_percentages):
        adjoints = [False, True] if network == 'odenet' else [False]
        for adjoint in adjoints:
            cmd = [
                'source', venv_path,
                '&&',  # To chain multiple commands
                'python', script_filename,
                '--network', network,
                '--adjoint', str(adjoint),
                '--batch_size', str(batch_size),
                '--data_percentage', str(data_percentage)
            ]

            # Run the command using subprocess
            subprocess.run(' '.join(cmd), shell=True)
