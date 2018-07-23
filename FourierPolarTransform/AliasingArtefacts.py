import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# x = np.linspace(0, 1024/60., 1024)
# y = 40*np.sin(60*np.pi*x)+150*np.sin(25*np.pi*x)+15*np.sin(18*np.pi*x)+240*np.sin(np.pi*x)
# fy = np.fft.fft(y)
#
# fx = np.linspace(0, len(x)/(x[-1] - x[0]), len(x)//2)                   # Frequency range
# plt.plot(fx, (np.abs(fy[:len(fy)/2]))*2./len(x))
#
# # fx = np.linspace(0, len(x)*2/(x[-1]-x[0]), len(x))
# # plt.plot(fx, np.abs(fy))
# plt.show()


x = np.linspace(-10, 10, 1024*5)
B = 80
y = B * np.sin(2*np.pi*B*x)/(np.pi * x) - (np.sin(np.pi * B * x)/(np.pi * x))**2
fy = np.fft.fft(y)

plt.plot(x, np.abs(np.fft.fftshift(fy)))
# plt.plot(x, y)
plt.show()


