import logging
from typing import Callable

import numpy as np
import torch.distributions
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader


class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.W(x)


class LinearDataSet(Dataset):
    def __init__(self, N: int):
        self.N = N
        in_out_dim = 2
        self.X = torch.distributions.Uniform(0, 1).sample(torch.Size([N, in_out_dim]))
        A = torch.tensor([[-1.0, 2.0], [1.0, -2.0]])
        self.Y = torch.einsum('bj,ji->bi', self.X, A.T)
        # sanity check
        A_lstsq = torch.linalg.lstsq(self.X, self.Y).solution.T
        x = 10

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
            loss = loss_fn(Y_hat, Y)
            loss.backward()
            optimizer.step()
            batches_losses.append(loss.item())
        epoch_avg_loss = np.nanmean(batches_losses)
        logger.info(f'Phase 1 : epoch {epoch},avg_loss = {epoch_avg_loss}')
        epoch += 1

    # second phase, LSTSQ-Augmented GD
    epoch = 0
    model_lstsq = Model(in_dim=2, out_dim=2)  # clone model
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
            with torch.no_grad():  # lets play it safe !
                weight_name = "W.weight"
                loss_gd = loss_fn(model(X), Y)
                # LSTSQ step
                W_lstsq = torch.linalg.lstsq(X, Y).solution.T
                model_lstsq.state_dict()[weight_name].data.copy_(W_lstsq)
                loss_lstsq = loss_fn(model_lstsq(X), Y)
                tot_u = loss_gd.item() + loss_lstsq.item()
                u_lstsq = loss_gd.item() / tot_u
                u_gd = loss_lstsq.item() / tot_u
                W_gd = model.state_dict()[weight_name].data
                W_new = u_gd * W_gd + u_lstsq * W_lstsq
                model.state_dict()[weight_name].data.copy_(W_new)
                loss_final = loss_fn(model(X), Y)
                # assert loss_final.item() < loss_gd.item()
                batches_losses.append(loss_final.item())

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
if __name__ == '__main__':
    epochs = int(1e8)  # approx. inf
    epochs1 = 10
    epochs2 = 1000
    lr = 1e-3
    N = 1000
    batch_size_1 = 64
    batch_size_2 = 128
    in_out_dim = 2
    loss_thr = 0.01
    #
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger()
    #
    model = Model(in_dim=in_out_dim, out_dim=in_out_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ds = LinearDataSet(N=N)
    data_loader_1 = DataLoader(dataset=ds, batch_size=batch_size_1, shuffle=True)
    data_loader_2 = DataLoader(dataset=ds, batch_size=batch_size_2, shuffle=True)
    loss_fn = MSELoss()
    # opt_gd(model=model, optimizer=optimizer, data_loader=data_loader,
    #        loss_fn=loss_fn, epochs=epochs, loss_thr=loss_thr)
    opt_gd_lstsq(model=model, optimizer=optimizer, data_loader1=data_loader_1, data_loader2=data_loader_2,
                 epochs1=epochs1,
                 epochs2=epochs2, loss_fn=loss_fn, loss_thr=loss_thr)
