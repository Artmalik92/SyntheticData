import numpy as np
import matplotlib.pyplot as plt


def c(t):
    return 2 * np.exp(0.3 * np.sin(t))


def simulated_seasonal_signals():
    outset = int(input("Enter outset: "))
    outcome = int(input("Enter outcome: "))
    a = float(input("Enter constant a: "))
    b = float(input("Enter constant b: "))
    d = float(input("Enter constant d: "))
    e = float(input("Enter constant e: "))

    num = (outcome - outset) * 365
    #data_length = num
    t = np.linspace(outset, outcome, num)
    #mjd_time = t.reshape(-1, 1)

    s = (a * np.sin(2 * np.pi * t) + b * np.cos(2 * np.pi * t) + c(t) * np.sin(2 * np.pi * t)
         + c(t) * np.cos(2 * np.pi * t) + d * np.sin(4 * np.pi * t) + e * np.cos(4 * np.pi * t))

    simulated_data = s

    return t, simulated_data


t, simulated_data = simulated_seasonal_signals()

plt.plot(t, simulated_data)
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title('Simulated Seasonal Signal')
plt.grid(True)
plt.show()