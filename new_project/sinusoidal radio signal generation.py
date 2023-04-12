import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the signal
frequency = 10 # Hz
amplitude = 2 # units
sampling_rate = 100 # Hz
duration = 1 # seconds

# Calculate the number of samples
num_samples = int(sampling_rate * duration)

# Generate the time axis
t = np.linspace(0, duration, num_samples)

# Generate the sinusoidal signal
signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Plot the signal
plt.plot(t, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (units)')
plt.title('Sinusoidal Radio Signal')
plt.show()