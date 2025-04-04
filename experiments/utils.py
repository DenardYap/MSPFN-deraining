import numpy as np
import cv2
# from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import kaiserord, firwin
from numpy.polynomial.chebyshev import Chebyshev
import matplotlib.pyplot as plt

def kaiser_lowpass_1d(pass_freq=0.45*np.pi,
                      stop_freq=0.55*np.pi,
                      ripple_db=0.5,
                      atten_db=40):
    pass_edge = pass_freq / np.pi
    stop_edge = stop_freq / np.pi
    transition_width = stop_edge - pass_edge
    
    N, beta = kaiserord(atten_db, transition_width)
    taps = firwin(N, pass_edge, window=('kaiser', beta), pass_zero=True)
    return taps

def chebyshev_T(n, x):
    c = np.zeros(n+1)
    c[n] = 1
    return Chebyshev(c)(x)

def mcclellan_transform_filter(shape, taps, A, B, C, D, E):
    M, N = shape
    H2d = np.zeros((M, N), dtype=np.float64)
    
    u_coords = 2.0 * np.pi * (np.arange(M) - M//2) / M
    v_coords = 2.0 * np.pi * (np.arange(N) - N//2) / N

    U, V = np.meshgrid(u_coords, v_coords, indexing='ij')

    Phi = A + B * np.cos(U) + C*np.cos(V) + D*np.cos(U - V) + E*np.cos(U + V)

    for n in range(len(taps)):
        H2d += taps[n] * chebyshev_T(n, Phi)

    return H2d

def build_8_direction_filters(shape, taps):
    param_sets = [
        (0, .5, -.5, 0, 0),
        (0, -.5, .5, 0, 0),
        (0, 0, 0, -.5, .5),
        (0, 0, 0, .5, -.5),
        (0, -.5, 0, 0, .5),
        (0, 0, -.5, .5, 0),
        (0, 0, -.5, 0, .5),
        (0, -.5, 0, .5, 0)
    ]
    filters = []
    for (A,B,C,D,E) in param_sets:
        H = mcclellan_transform_filter(shape, taps, A, B, C, D, E)
        where = np.where(H < 0.001)
        H[where] = 0
        kernel = np.ones((10, 10), np.float64)
        H = cv2.dilate(H, kernel)
        H = cv2.GaussianBlur(H, (5, 5), 0)
        filters.append(H)

    w0 = filters[0]*filters[2] #0
    w1 = filters[0]*filters[3] #1
    w2 = filters[1]*filters[2] #2
    w3 = filters[1]*filters[3] #3

    f0 = w2*filters[6]
    f1 = w0*filters[4]
    f2 = w1*filters[7]
    f3 = w3*filters[5]
    f4 = w2 - f0
    f5 = w0 - f1
    f6 = w1 - f2
    f7 = w3 - f3

    # I messed up the numbering of the filters and I am way too tired to thnk
    # about how to switch it so here is the map of indixes to my (out of order)
    # filter numbers 
    #0: 4
    #1: 0
    #2: 1
    #3: 5
    #4: 2
    #5: 6
    #6: 3
    #7: 7
    eight_directions = [1-f4, 1-f0, 1-f1, 1-f5, 1-f2, 1-f6, 1-f3, 1-f7]

    return eight_directions


def img_float64_2_uint8(image_float64):
    image_normalized = (image_float64 - np.min(image_float64)) / (np.max(image_float64) - np.min(image_float64))
    image_uint8 = (image_normalized * 255).astype(np.uint8)
    return image_uint8