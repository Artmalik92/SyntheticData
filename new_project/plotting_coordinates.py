import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

'''
# подключение к БД
cnx = sqlite3.connect('D:/docs_for_univer/diplom/fags_ppp_timeseries (1).sqlite3')
df = pd.read_sql_query("SELECT * FROM mark_coordinate", cnx)
'''

'''
# Параметры
years = 2                           # кол-во лет
frequency = 365                     # Частота (Гц)
amplitude = -2.97770129341665e-03   # Амплитуда
sampling_rate = 1                   # Выборка (units)
duration = 365 * years              # Длительность (days)
white_noise_level = 0.005           # Уровень белого шума
flicker_noise_level = 0.001         # Уровень фликер шума
alpha = 0.01                        # Экспонента степенного закона
X0 = 4.52260828455453e+05
VX = -2.57928156083501e-02

# Рассчет кол-ва выборок
num_samples = int(sampling_rate * duration)

# Временная ось
t = np.linspace(0, duration, num_samples)

# Движение литосферной плиты
lithosphere_signal = X0+VX*t/365

# Синусоидальный сигнал
sinus_signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Добавляем смещение литосферной плиты
signal = sinus_signal + lithosphere_signal


# Степенной закон
power_law = np.random.randn(num_samples) / (t + 1)**(alpha/2.0)

# Добавление степенного закона в сигнал
total_signal = signal #+ power_law

# Добавление белого шума
white_noise = white_noise_level * np.random.randn(num_samples)
synthetic_data = total_signal + white_noise

# Фликер-шум
flicker_noise = flicker_noise_level * np.random.randn(num_samples) * np.sqrt(1/t)

synthetic_data = synthetic_data + flicker_noise

'''
def show_synthetic_data():
    df = pd.read_csv('data.csv', delimiter=',')
    print(df)

    df = df.set_index('Station')
    df_NSK1 = df.loc['NSK1']
    df_NSK1 = df_NSK1.reset_index()
    df_NSK1_X = df_NSK1['X']
    df_NSK1_Y = df_NSK1['Y']
    df_NSK1_Z = df_NSK1['Z']

    # График составляющих сигнала
    plt.figure(1)
    plt.plot(t, white_noise, label='white noise')
    plt.plot(t, flicker_noise, label='flicker noise')
    plt.xlabel('Time (days)')
    plt.ylabel('Amplitude (units)')
    plt.title('Компоненты сигнала')
    plt.legend()

    # "Чистый" сигнал
    plt.figure(2)
    plt.plot(t, sinus_signal, label='signal')
    plt.xlabel('Time (days)')
    plt.ylabel('Amplitude (units)')
    plt.title('"Чистый" сигнал')
    plt.legend()

    # Конечный сигнал
    plt.figure(3)
    plt.plot(t, synthetic_data, label='synthetic data')
    plt.xlabel('Time (days)')
    plt.ylabel('Amplitude (units)')
    plt.title('Конечный сигнал')
    plt.legend()

    plt.figure(4)
    plt.plot(t, lithosphere_signal, label='lithosphere_signal')
    plt.xlabel('Time (days)')
    plt.ylabel('Amplitude (units)')
    plt.title('Движение плиты')
    plt.legend()

    '''
    # Степенной закон
    plt.figure(4)
    plt.plot(t, power_law, label='power law')
    plt.xlabel('Time (days)')
    plt.ylabel('Amplitude (units)')
    plt.title('Степенной закон')
    '''
    plt.show()


def nsk1_test(file, station_name):
    df = pd.read_csv(file, delimiter=';')
    df = df.set_index('Station')
    df_NSK1 = df.loc[station_name]
    df_NSK1 = df_NSK1.reset_index()
    df_NSK1_X = df_NSK1['X']
    df_NSK1_Y = df_NSK1['Y']
    df_NSK1_Z = df_NSK1['Z']

    plt.figure(1)
    df_NSK1_X.plot()
    plt.xlabel('Weeks')
    plt.ylabel('Amplitude')
    plt.title('NSK1 X-coordinate')

    plt.figure(2)
    df_NSK1_Y.plot()
    plt.xlabel('Weeks')
    plt.ylabel('Amplitude')
    plt.title('NSK1 Y-coordinate')

    plt.figure(3)
    df_NSK1_Z.plot()
    plt.xlabel('Weeks')
    plt.ylabel('Amplitude')
    plt.title('NSK1 Z-coordinate')

    plt.show()


#nsk1_test('Data/data_c0b3_XYZ_harmonics(Y)_linear(Y)_noise(Y)_impulse(Y).csv', 'NSK1')

