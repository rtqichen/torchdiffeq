import argparse
import logging
import pickle

import numpy as np
import torch

from examples.models import HyperNetwork, CNF

# get logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Global-vars
N_SAMPLES = 1024

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--trajectory-opt', type=str, choices=['vanilla', 'hybrid'])
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# device
device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    meta_data = pickle.load(open(f'artifacts/{args.trajectory_opt}_{args.version}.pkl', "rb"))
    dim = meta_data['dim']
    hidden_dim = meta_data['args']['hidden_dim']
    width = meta_data['args']['width']
    trajectory_model = CNF(in_out_dim=dim, hidden_dim=hidden_dim, width=width,device=device)
    trajectory_model.load_state_dict(meta_data['model'])

    # 1. verify normality of generated data
    ## Generate samples out of the loaded model and meta-data
    z0 = meta_data['base_distribution'].sample(torch.Size([N_SAMPLES])).to(device)
    logp_diff_t0 = torch.zeros(N_SAMPLES, 1).type(torch.float32).to(device)
    t0 = meta_data['args']['t0']
    tN = meta_data['args']['t1']
    t_vals = torch.tensor(list(np.arange(t0, tN + 1, 1)))
    u = odeint(func=trajectory_model, y0=(z0,logp_diff_t0), t=t_vals)
    x=10
    # # 2. verify parameters of generated data
    # logger.info(f'Sub-experiment finished')

