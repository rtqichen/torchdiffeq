import logging
from argparse import ArgumentParser
from typing import Callable

import numpy as np
import torch.distributions
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing, load_diabetes

from examples.torch_rbf import RBFN


class NN2LayerModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)
        self.activation = torch.nn.Tanh()
        self.net = torch.nn.Sequential(self.linear1, self.activation, self.linear2)

    def forward(self, x):
        return self.net(x)

    def forward2(self, x):
        y1 = self.linear1(x)
        y2 = self.activation(y1)
        return y2


class PolyTrigActiv(Dataset):
    def __init__(self, N: int, pow: int):
        self.N = N
        self.in_dim = 2
        self.out_dim = 2
        eps = 1e-4
        self.X = torch.distributions.Uniform(0, 1).sample(torch.Size([N, self.in_dim]))
        A = torch.tensor([[-1.0, 2.0], [1.0, -2.0]])
        X_non_linear = torch.sin(torch.exp(self.X ** pow)) + torch.tanh(torch.exp(self.X ** pow))
        self.Y = torch.einsum('bj,ji->bi', X_non_linear, A.T)
        # sanity check
        A_lstsq = torch.linalg.lstsq(X_non_linear, self.Y).solution.T
        assert torch.norm(A - A_lstsq).item() <= eps

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class DiabetesDataset(Dataset):
    def __init__(self):
        X_raw, Y = load_diabetes(return_X_y=True)
        scaler = StandardScaler()
        scaler.fit(X_raw)
        X = scaler.transform(X_raw)
        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
        self.N = self.X_tensor.size()[0]
        self.in_dim = self.X_tensor.size()[1]
        self.out_dim = self.Y_tensor.size()[1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.Y_tensor[idx]


class CaliforniaDataset(Dataset):
    def __init__(self):
        data = fetch_california_housing()
        scaler = StandardScaler()
        X_raw = data.data
        Y = data.target
        scaler.fit(X_raw)
        X = scaler.transform(X_raw)
        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
        self.N = self.Y_tensor.size()[0]
        self.in_dim = self.X_tensor.size()[1]
        self.out_dim = self.Y_tensor.size()[1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.Y_tensor[idx]


class PolyDataSet(Dataset):
    def __init__(self, N: int):
        self.N = N
        self.in_dim = 2
        self.out_dim = 2
        eps = 1e-4
        pow = 3
        self.X = torch.distributions.Uniform(0, 1).sample(torch.Size([N, self.in_dim]))
        A = torch.tensor([[-1.0, 2.0], [1.0, -2.0]])
        self.Y = torch.einsum('bj,ji->bi', torch.pow(self.X, pow), A.T)
        # sanity check
        A_lstsq = torch.linalg.lstsq(torch.pow(self.X, pow), self.Y).solution.T
        assert torch.norm(A - A_lstsq).item() <= eps

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def opt_gd(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
           data_loader: DataLoader, epochs: int, loss_fn: Callable, loss_thr: float):
    train_method_str = "GD"
    logger = logging.getLogger()
    batches_losses = []
    epoch = 0
    epoch_avg_loss = np.inf
    while epoch <= epochs:
        for i, (X, Y) in enumerate(data_loader):
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)
            loss.backward()
            optimizer.step()
            batches_losses.append(loss.item())
        epoch_avg_loss = np.nanmean(batches_losses)
        logger.info(f'epoch {epoch},avg_loss = {epoch_avg_loss}')
        epoch += 1
        if epoch_avg_loss <= loss_thr:
            logger.info(f'At epoch = {epoch} epoch_avg_loss = '
                        f'{epoch_avg_loss} <= loss_thr = {loss_thr},quitting !')
            break
    logger.info(f'Training Finished using {train_method_str} '
                f'at epoch = {epoch} with loss = {epoch_avg_loss}')


def opt_gd_lstsq(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 data_loader1: DataLoader, data_loader2,
                 epochs1: int, epochs2, loss_fn: Callable, loss_thr: float):
    train_method_str = "GD-LSTSQ"

    logger = logging.getLogger()
    batches_losses = []
    epoch = 0
    epoch_avg_loss = np.inf
    # phase1
    while epoch <= epochs1:
        for i, (X, Y) in enumerate(data_loader1):
            optimizer.zero_grad()
            Y_hat = model(X)
            # Y2= model.forward2(X) # fixme, just for debugging
            loss = loss_fn(Y_hat, Y)
            loss.backward()
            optimizer.step()
            batches_losses.append(loss.item())
        epoch_avg_loss = np.nanmean(batches_losses)
        logger.info(f'Phase 1 : epoch {epoch},avg_loss = {epoch_avg_loss}')
        epoch += 1
    # fixme, for debugging
    # for name, param in model.named_parameters():
    #     print(f'{name} = {param}')
    # second phase, LSTSQ-Augmented GD
    logger.info("===")
    epoch = 0
    while epoch < epochs2:
        batches_losses = []
        for i, (X, Y) in enumerate(data_loader2):
            # GD Step
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)
            loss.backward()
            optimizer.step()
            # Augmented update
            # TODO make it a method
            # if True and epoch > 10 and epoch % 10 == 0:
            # logger.info('lstsq kicked in')
            with torch.no_grad():  # lets play it safe !
                weight_name = "linear2.weight"
                bias_name = "linear2.bias"
                loss_gd = loss_fn(model(X), Y)
                W_gd = torch.clone(model.state_dict()[weight_name].data)
                b_gd = torch.clone(model.state_dict()[bias_name].data)
                # LSTSQ step
                Y2 = model.forward2(X)
                b_size = Y2.size()[0]
                Y2 = torch.cat([Y2, torch.ones(b_size).view(-1, 1)], dim=1)
                try:
                    Wnb_lstsq = torch.linalg.lstsq(Y2, Y).solution.T
                except Exception as e:
                    x = 10
                W_lstsq = Wnb_lstsq[:, :model.hidden_dim]
                b_lstsq = Wnb_lstsq[:, model.hidden_dim:]

                model.state_dict()[weight_name].data.copy_(W_lstsq)
                model.state_dict()[bias_name].data.copy_(b_lstsq.flatten())
                loss_lstsq = loss_fn(model(X), Y)
                tot_u = loss_gd.item() + loss_lstsq.item()
                # u_lstsq =  loss_gd.item() / tot_u
                # u_gd = loss_lstsq.item() / tot_u
                if loss_lstsq.item() < loss_gd.item():
                    W_new = W_lstsq
                    b_new = b_lstsq.view(-1)
                else:
                    W_new = W_gd
                    b_new = b_gd.view(-1)
                model.state_dict()[weight_name].data.copy_(W_new)
                model.state_dict()[bias_name].data.copy_(b_new)
                loss_final = loss_fn(model(X), Y)
                # assert loss_final.item() < loss_gd.item()
                batches_losses.append(loss_final.item())

            batches_losses.append(loss.item())

        epoch_avg_loss = np.nanmean(batches_losses)
        logger.info(f'Phase 2 : epoch {epoch},avg_loss = {epoch_avg_loss}')
        epoch += 1
        if epoch_avg_loss <= loss_thr:
            logger.info(f'At Phase2 : epoch = {epoch} epoch_avg_loss = '
                        f'{epoch_avg_loss} <= loss_thr = {loss_thr},quitting !')
            break

    logger.info(f'Training Finished using {train_method_str} '
                f'at total-epoch = {epoch + epochs1} with loss = {epoch_avg_loss}')


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)5s() ] %(message)s"


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--opt-method", type=str, required=True, choices=["gd", "lstsq-gd"], help='opt method')
    return parser


if __name__ == '__main__':
    epochs = 1000
    epochs1 = 300
    epochs2 = 300

    assert (epochs1 + epochs2) <= epochs  # to bring value
    lr = 1e-3
    N = 1000
    batch_size_1 = 64
    batch_size_2 = 128
    loss_thr = 0.01
    hidden_dim = 20
    pow =3
    #
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger()
    #
    parser = get_parser()
    args = parser.parse_args()
    ## Data
    # data_set = PolyDataSet(N=N)
    # data_set = CaliforniaDataset()
    # data_set = DiabetesDataset()
    data_set = PolyTrigActiv(N=N,pow=6)
    assert hasattr(data_set, "in_dim")
    assert hasattr(data_set, "out_dim")
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size_1, shuffle=True)
    data_loader_1 = DataLoader(dataset=data_set, batch_size=batch_size_1, shuffle=True)
    data_loader_2 = DataLoader(dataset=data_set, batch_size=batch_size_2, shuffle=True)

    ## Model
    # model = LinearModel(in_dim=in_out_dim, out_dim=in_out_dim)
    # model = NN2LayerModel(in_dim=data_set.in_dim, out_dim=data_set.out_dim, hidden_dim=hidden_dim)
    model = RBFN(in_dim=data_set.in_dim, n_centers=hidden_dim, out_dim=data_set.out_dim,
                 basis_fn_str="gaussian")
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    #
    print(f'Model = {model}')
    print(f'Dataset = {data_set}')
    if args.opt_method == "gd":
        opt_gd(model=model, optimizer=optimizer, data_loader=data_loader,
               loss_fn=loss_fn, epochs=epochs, loss_thr=loss_thr)
    elif args.opt_method == "lstsq-gd":
        opt_gd_lstsq(model=model, optimizer=optimizer, data_loader1=data_loader_1,
                     data_loader2=data_loader_2, epochs1=epochs1,
                     epochs2=epochs2, loss_fn=loss_fn, loss_thr=loss_thr)
