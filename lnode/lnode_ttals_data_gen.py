import argparse
import logging
import pickle

import torch

# get logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    artifacts = pickle.load(open(f'artifacts/lnode_{args.timestamp}.pkl', "rb"))
    trajectory_model = torch.load(f'artifacts/trajectory_model_{args.timestamp}.model')
    x = 10
