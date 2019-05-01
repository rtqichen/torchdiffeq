import matplotlib.pyplot as plt
import numpy as np

# Shapes:   Snake,
#           trefoil,
#           rose (k=4, k=7),
#           torus (3 params),
#           spiral,
#           logspiral,
#           spirograph
def snake(t, f = 2):
    # f: Number of oscillations per 2pi*t travelled
    return np.array([np.cos(f*t), t])

def trefoil(t):
    return np.array([np.sin(t) + 2*np.sin(2*t),
                     np.cos(t) - 2*np.cos(2*t)])

def rose(t, k = 6):
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
    samples = f(np.arange(t0, t1, dt))
    samples = np.moveaxis(samples, 0, 1)
    return samples

def vis(samples):
    plt.figure()
    plt.plot(samples[:,0], samples[:,1])
    plt.show()

def vg(f):
    vis(gen(f))

def gspirograph(a = 11, b = 6):
    t= np.arange(-50, 50, 0.01)
    x= a*np.cos(t) - b*np.cos(a*t/b)
    y= a*np.sin(t) - b*np.sin(a*t/b)
    return np.stack((x, y), axis=1)
