from torch.distributions import MultivariateNormal, Normal
import torch


def my_log_prob(x: torch.Tensor, mio: float, sigma: float):
    exponent = -0.5 * ((x - mio) / sigma) ** 2
    prob = 1 / (torch.sqrt(2 * torch.tensor(torch.pi)) * sigma) * torch.exp(torch.tensor(exponent))
    log_prob = torch.log(prob)
    return prob, log_prob


if __name__ == '__main__':
    mio = -1.0
    sigma = 2.0
    n = 5000
    # mvn = MultivariateNormal(loc=mio, covariance_matrix=sigma)
    ndist = Normal(loc=mio, scale=sigma)
    samples = ndist.sample(sample_shape=torch.Size([n, ]))
    log_prob = ndist.log_prob(value=samples)
    prob, log_prob2 = my_log_prob(x=samples, mio=mio, sigma=sigma)
    print('my log calc error')
    print(torch.norm(log_prob2-log_prob))
    print('log prob mean')
    print(log_prob2.mean(0))
