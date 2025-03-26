import numpy as np
# import cv2
# from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import kaiserord, firwin
from numpy.polynomial.chebyshev import Chebyshev

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
        (0.2, 0.2, -0.2, 0.2, -0.2),
        # (0.0, 0.5, 0.5, 0.0, 0.0),
        # (0.0, -0.5, 0.5, 0.0, 0.0),
        # (0.0, 0.0, 0.0, -0.5, 0.5),
        # (0.0, 0.0, 0.0, 0.5, -0.5),
        # (0.0, -0.5, 0.0, 0.0, 0.5),
        # (0.0, 0.0, -0.5, 0.5, 0.0),
        # (0.0, 0.0, -0.5, 0.0, 0.5),
        # (0.0, 0.5, 0.0, 0.0, -0.5),
    ]
    filters = []
    for (A,B,C,D,E) in param_sets:
        H = mcclellan_transform_filter(shape, taps, A, B, C, D, E)
        filters.append(H)
    return filters
