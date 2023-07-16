from torch.distributions import MultivariateNormal, Normal,Uniform
import torch
import pingouin as pg


def my_log_prob(x: torch.Tensor, mio: float, sigma: float):
    exponent = -0.5 * ((x - mio) / sigma) ** 2
    prob = 1 / (torch.sqrt(2 * torch.tensor(torch.pi)) * sigma) * torch.exp(torch.tensor(exponent))
    log_prob = torch.log(prob)
    return prob, log_prob


if __name__ == '__main__':
    mio = -3.0
    sigma = 2.0
    n = 30000
    # mvn = MultivariateNormal(loc=mio, covariance_matrix=sigma)
    # ndist = Normal(loc=mio, scale=sigma)
    mio = torch.tensor([-2.0, 3.0])
    sigma = torch.diag(torch.tensor([0.2, 0.8]))
    mvn = MultivariateNormal(loc=mio, covariance_matrix=sigma)
    samples = mvn.sample(sample_shape=torch.Size([n, ]))
    samples = Uniform(low=0.1,high=0.9).sample(torch.Size([n,2]))
    # log_prob = ndist.log_prob(value=samples)
    # prob, log_prob2 = my_log_prob(x=samples, mio=mio, sigma=sigma)
    #print('my log calc error')
    #print(torch.norm(log_prob2 - log_prob))
    #print('log prob mean')
    # print(log_prob2.mean(0))
    #print(samples.mean(0))
    #print(torch.std(samples))

    # https://pingouin-stats.org/build/html/generated/pingouin.multivariate_normality.html
    samples_np = samples.detach().numpy()
    t = pg.multivariate_normality(X=samples_np)
    print(t)
