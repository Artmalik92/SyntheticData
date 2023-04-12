import numpy as np
from numpy.linalg import pinv
import math
from scipy.stats import chi2
import pandas as pd

Data = pd.read_csv('data.csv', delimiter=',')

df = Data.set_index('Station')
df_NSK1 = df.loc['NSK1']
df_NSK1 = df_NSK1.reset_index()
df_NSK1_X = df_NSK1['X']
df_NSK1_Y = df_NSK1['Y']
df_NSK1_Z = df_NSK1['Z']


# массив координат пунктов на начальную эпоху
net_0 = (447670.300, 3638117.390, 5202281.560,
         -178337.950, 3567385.910, 5266736.150,
         -307566.910, 3947896.900, 4984000.200)

# скорости пунктов
velocities1 = (-0.0282, 0.0048, 0.0024)
velocities2 = (-0.0248, 0.0022, -0.0005)
velocities3 = (-0.0211, 0.0032, 0.0015)

# кол-во лет между эпохами
epoch = 5

# массив координат пунктов на n-ую эпоху
net_i = (net_0[0]-velocities1[0]*epoch, net_0[1]-velocities1[1]*epoch, net_0[2]-velocities1[2]*epoch,
         net_0[3]-velocities2[0]*epoch, net_0[4]-velocities2[1]*epoch, net_0[5]-velocities2[2]*epoch,
         net_0[6]-velocities3[0]*epoch, net_0[7]-velocities3[1]*epoch, net_0[8]-velocities3[2]*epoch)

skp = None

# уровень значимости для статистического теста
significance_level = 0.05

# статус гипотезы хи-квадрат
hypothesis_status = 1


def main():
    while hypothesis_status == 1:
        pass


# расчет базовой линии
def baseline(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# основная часть
def congruence_test(net_0, net_i, significance_level):
    line12 = baseline(net_0[0], net_0[1], net_0[2], net_0[3], net_0[4], net_0[5])
    line23 = baseline(net_0[3], net_0[4], net_0[5], net_0[6], net_0[7], net_0[8])
    line31 = baseline(net_0[6], net_0[7], net_0[8], net_0[0], net_0[1], net_0[2])

    line12_i = baseline(net_i[0], net_i[1], net_i[2], net_i[3], net_i[4], net_i[5])
    line23_i = baseline(net_i[3], net_i[4], net_i[5], net_i[6], net_i[7], net_i[8])
    line31_i = baseline(net_i[6], net_i[7], net_i[8], net_i[0], net_i[1], net_i[2])

    difference = (line12 - line12_i, line23 - line23_i, line31 - line31_i)

    d = np.array(difference)

    Qdd = np.eye(3)

    sigma_0 = 0.005

    K = d.transpose().dot(pinv(Qdd)).dot(d)/(sigma_0**2)

    # тестовое значение функции хм квадрат при заданном уровне доверительной вероятности 0.05
    test_value = chi2.ppf(df=d.shape[0], q=significance_level)

    if K > test_value:
        print('null hipotesis is rejected')
        hypothesis_status = 1
    else:
        print('null hipotesis is not rejected ')
        hypothesis_status = 0

    return K


print(congruence_test(net_0, net_i, significance_level))

