import numpy as np
import matplotlib.pyplot as plt

# Параметры
frequency = 10                  # Частота (Гц)
amplitude = 2                   # Амплитуда
sampling_rate = 100             # Выборка (Гц)
duration = 1                    # Длительность (сек)
white_noise_level = 0.1         # Уровень белого шума
flicker_noise_level = 0.05      # Уровень фликер шума
alpha = 2.0                     # Экспонента степенного закона


# Рассчет кол-ва выборок
num_samples = int(sampling_rate * duration)

# Временная ось
t = np.linspace(0, duration, num_samples)

# Синусоидальный сигнал
signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Степенной закон
power_law = np.random.randn(num_samples) / (t + 1)**(alpha/2.0)

# Добавление степенного закона в сигнал
total_signal = signal + power_law

# Добавление белого шума
white_noise = white_noise_level * np.random.randn(num_samples)
synthetic_data = total_signal + white_noise

# Фликер-шум
flicker_noise = flicker_noise_level * np.random.randn(num_samples) * np.sqrt(1/t)

synthetic_data = synthetic_data + flicker_noise

# График составляющих сигнала
plt.figure(1)
plt.plot(t, signal, label='signal')
plt.plot(t, white_noise, label='white noise')
plt.plot(t, flicker_noise, label='flicker noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (units)')
plt.title('Компоненты сигнала')
plt.legend()

# Конечный сигнал
plt.figure(2)
plt.plot(t, synthetic_data, label='synthetic data')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (units)')
plt.title('Конечный сигнал')
plt.legend()


plt.figure(3)
plt.plot(t, power_law, label='power law')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (units)')
plt.title('Степенной закон')


plt.show()

