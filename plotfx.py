import matplotlib.pyplot as plt
import numpy as np

def snake():
    yvals= np.arange(0, 20, 0.01) #produces 2000 points
    xvals = np.cos(yvals)
    return np.stack((xvals, yvals), axis=1)

def trefoil():
    tvals= np.arange(-10, 10, 0.01) #produces 2000 points
    xvals= np.sin(tvals) + 2*np.sin(2*tvals)
    yvals= np.cos(tvals) - 2*np.cos(2*tvals)
    return np.stack((xvals, yvals), axis=1)

def rose(k = 2):
    tvals= np.arange(-10, 10, 0.01) #2000 points
    xvals= np.cos(k*tvals)*np.cos(tvals)
    yvals= np.cos(k*tvals)*np.sin(tvals)
    return np.stack((xvals, yvals), axis=1)

def torus(p = 2, q = 3):
    phi= np.arange(0, 2*np.pi, 0.01) #629 points
    r= np.cos(q*phi) + 2
    x= r*np.cos(p*phi)
    y= r*np.sin(p*phi)
    return np.stack((x, y), axis=1)

def spiral():
    tvals= np.arange(0, 6*np.pi, 0.01) #1885 points
    r= tvals**2
    x= r*np.cos(tvals)
    y= r*np.sin(tvals)
    return np.stack((x, y), axis=1)

def spirograph(a = 11, b = 6):
    t= np.arange(-50, 50, 0.01)
    x= a*np.cos(t) - b*np.cos(a*t/b)
    y= a*np.sin(t) - b*np.sin(a*t/b)
    return np.stack((x, y), axis=1)
