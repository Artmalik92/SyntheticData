import pandas as pd
import numpy as np
from scipy.stats import t


df = pd.read_csv('Data/test_file_xyz_with_noise.csv', delimiter=';')


def baseline():
