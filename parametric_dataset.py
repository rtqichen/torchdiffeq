"""parametric_dataset

Provides:
    Parametric shape functions f(t) --> (x,y)
        Where t = int, float, or 1D np.array
     - snake, trefoil, rose, torus, spiral, logspiral
    gen(f, t0 = 0, t1 = 2*np.pi, dt = 0.001, noise_std = 0)
    vis(sample)
    vg(f)
    R(deg, use_radians)
    rotate(sample, deg)
    translate(sample, T)
"""
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin

def snake(t, f = 2):
    # f: Number of oscillations per 2pi*t travelled
    return np.array([np.cos(f*t), t])

def trefoil(t):
    return np.array([np.sin(t) + 2*np.sin(2*t),
                     np.cos(t) - 2*np.cos(2*t)])

def rose(t, k = 7):
    # k = number of petals if odd, else = number of petals / 2
    return np.array([np.cos(k*t)*np.cos(t),
                     np.cos(k*t)*np.sin(t)])

def torus(t, p=2, q=3):
    # torus knot, parameterized by p and q
    # Interesting shapes:
    # p = 3, q = 2
    # p = 3, q = 4
    # p = 2, q = 3  
    # p = 7, q = 13
    r = r= np.cos(q*t) + 2
    return np.array([r*np.cos(p*t), r*np.sin(p*t)])

def spiral(t, r=3):
    # a = scale
    # r = rate of increase.
    # Alternatively, number of spirals in t=0 to t=2*pi
    # radius = theta
    # angle = theta
    return np.array([r*t*np.cos(r*t), r*t*np.sin(r*t)])

def logspiral(t, a=1, b=.1759, r = 3):
    # r = increases the speed of time
    # a = scale
    # b = speed of exponential growth
    l = a*np.exp(b*r*t)
    return np.array([l*np.cos(r*t), l*np.sin(r*t)])

def spirograph(t, a = 11, b = 6):
    # a, b are spirograph parameters
    # See: 11,6, 17,3
    return np.array([a*np.cos(t*b) - b*np.cos(a*t),
                     a*np.sin(t*b) - b*np.sin(a*t)])


def gen(f, t0 = 0, t1 = 2*np.pi, dt = 0.001, noise_std = 0):
    '''gen
    # Arguments:
    f: function from t --> (x,y) pairs. (See snake, trefoil, etc. above)
    t0: number, starting time value
    t1: number, ending time value
    dt: number, resolution of data. (smaller dt = more data, higher resolution.)
    noise_std: number >= 0, std of gaussian noise to add to data points.
    '''
    # From a shape f(t), generate samples.
    sample = f(np.arange(t0, t1, dt))
    sample = np.moveaxis(sample, 0, 1)
    if noise_std:
        return np.random.gaussian(sample, scale=noise_std)
    return sample


def vis(sample):
    # Convenience function for plotting a sample
    plt.figure()
    plt.plot(sample[:,0], sample[:,1])
    plt.show()

def vg(f):
    # Convenience function for visualizing a shape
    vis(gen(f))

# Augmentors
def R(deg, use_radians=False):
    # Get rotation matrix in radians
    if use_radians:
        return np.array([[np.cos(deg), -np.sin(deg)],
                         [np.sin(deg),  np.cos(deg)]])
    return R(deg*0.01745329251993889, use_radians=True)

def rotate(sample, deg):
    return np.matmul(sample, R(deg))

def translate(sample, T):
    return sample + T


def generate_sample(res = 1000, rotation_augment = True, trans_augment_std = 1):
    """Generate a single training example.
    Randomly chooses a shape and augments its data.
    
    res: Integer, the number of datapoints to plot.
    rotation_augment: Boolean; if true, rotate the data by 360 degrees randomly.
    trans_augment_std: The std of the translation to be applied.
    """
    # res: number of datapoints in the sample
    # rotation_augment = True, rotate randomly
    # trans_augment_std: translation augmentation standard deviation, after normalization
    
    # Randomly generate one true sample, with random augmentations.
    shape_functions = [snake,
                       trefoil,
                       lambda t : rose(t,k=7),
                       lambda t : rose(t,k=2),
                       lambda t : torus(t,p=1,q=6),
                       lambda t : torus(t,p=2,q=3),
                       lambda t : torus(t,p=3,q=2),
                       spiral,
                       logspiral,
                       spirograph]
    # Choose f randomly, acquire 'res' datapoints
    f = shape_functions[np.random.randint(len(shape_functions)-1)]
    data_sample = gen(f, dt = 2*np.pi/res)
    # Rotation augment
    if rotation_augment:
        data_sample = rotate(data_sample, deg=np.random.rand()*360)
    # Scale such that maximum dimension of bounding box is 1
    xmin, ymin = np.min(data_sample, axis=0)
    xmax, ymax = np.max(data_sample, axis=0)
    scale_factor = max(ymax-ymin, xmax-xmin)
    data_sample = data_sample / scale_factor
    # Translation augment
    data_sample = data_sample + [np.random.normal(scale=trans_augment_std), np.random.normal(scale=trans_augment_std)]
    return data_sample

# Generate our dataset...
def generate_true_dataset(nsamples=10000, rotation_augment = True, trans_augment_std = .2):
    return np.stack([generate_sample(rotation_augment = rotation_augment, 
                                     trans_augment_std = trans_augment_std)
                     for _ in range(nsamples)])

#dataset = generate_true_dataset()
#np.save("parametric_dataset_10000_samples", dataset)
#dataset = np.load("parametric_dataset_10000_samples.npy")
# Take 500 samples samples, + noise with 1/128 std dev
#train_dataset = np.random.normal(dataset[:,200:700,:], scale=2**-7)
#np.save("train_parametric_dataset", train_dataset)

# Generate images to visualize
shape_functions = [snake,
                   trefoil,
                   lambda t : rose(t,k=7),
                   lambda t : rose(t,k=2),
                   lambda t : torus(t,p=1,q=6),
                   lambda t : torus(t,p=2,q=3),
                   lambda t : torus(t,p=3,q=2),
                   spiral,
                   logspiral,
                   spirograph]

[vg(f) for f in shape_functions]
