from sklearn.datasets import make_gaussian_quantiles,make_circles
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html
    x, y = make_gaussian_quantiles(n_classes=3, n_features=3, n_samples=1000, shuffle=True)
    # x , _ = make_circles(n_samples=1000)

    plt.scatter(x[:, 0], x[:, 1],c=y)
    plt.savefig('fig.png')
    x = 10
