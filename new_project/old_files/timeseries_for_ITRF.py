from pandas import DataFrame
import math as m
import numpy as np
from numpy import random

def test_coordinate_time_series_generator(period):
    x0 = 4.52260828455453e+05
    y0 = 3.63587749472371e+06
    z0 = 5.20345321431246e+06
    vx = -2.57928156083501e-02
    vy = 2.25331944516226e-03
    vz = -9.43054722599818e-04
    X_cosine_annual = -2.97770129341665e-03
    X_sine_annual = -1.52824655969895e-03
    Y_cosine_annual = -7.50343793311789e-03
    Y_sine_annual = -2.93987584804527e-03
    Z_cosine_annual = -1.13996274591522e-02
    Z_sine_annual = -5.53152026612332e-03
    X_cosine_semiannual = -2.97770129341665e-03
    X_sine_semiannual = -1.52824655969895e-03
    Y_cosine_semiannual = -7.50343793311789e-03
    Y_sine_semiannual = -2.93987584804527e-03
    Z_cosine_semiannual = -1.13996274591522e-02
    Z_sine_semiannual = -5.53152026612332e-03
    coordinate_time_series = np.zeros((period, 4))

    for t in range(period):
        xt = x0 + vx * t / 365 + X_cosine_annual * m.cos(2 * m.pi * t / 365) + X_sine_annual * m.sin(2 * m.pi * t / 365) + X_cosine_semiannual * m.cos(4 * m.pi * t / 365) + X_sine_semiannual * m.sin(4 * m.pi * t / 365) + random.randint(10, 10) / 10000
        yt = y0 + vy * t / 365 + Y_cosine_annual * m.cos(2 * m.pi * t / 365) + Y_sine_annual * m.sin(2 * m.pi * t / 365) + Y_cosine_semiannual * m.cos(4 * m.pi * t / 365) + Y_sine_semiannual * m.sin(4 * m.pi * t / 365) + random.randint(10, 10) / 10000
        zt = z0 + vz * t / 365 + Z_cosine_annual * m.cos(2 * m.pi * t / 365) + Z_sine_annual * m.sin(2 * m.pi * t / 365) + Z_cosine_semiannual * m.cos(4 * m.pi * t / 365) + Z_sine_semiannual * m.sin(4 * m.pi * t / 365) + random.randint(10, 10) / 10000
        coordinate_time_series[t][0] = xt
        coordinate_time_series[t][1] = yt
        coordinate_time_series[t][2] = zt
        coordinate_time_series[t][3] = t
    a=DataFrame(coordinate_time_series, columns= ['X','Y','Z','Epoch'])

    return a

print(test_coordinate_time_series_generator(365))

