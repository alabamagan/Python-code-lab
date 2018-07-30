import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

N = 1024.       # Sampled points
s = 64.         # Sampling rate
R = N/s         # x-space range

x = np.linspace(0, R, N)    # x-space steps
y = 40*np.sin(60*np.pi*x)+150*np.sin(25*np.pi*x)+15*np.sin(18*np.pi*x)  # Signal
fy = np.fft.fft(y)

fx = np.linspace(0, s, len(x)//2)                   # Frequency range
plt.plot(fx, (np.abs(fy[:len(fy)/2]))*2./N)         # Integration is normalize by dividing half of number of points

plt.show()



