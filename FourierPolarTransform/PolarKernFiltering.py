from Algorithms.PolarTransform import cart2pol_2d, pol2cart_2d
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2, ifft2
from scipy.ndimage import imread
from scipy.signal import convolve2d

def main():
    # Generate an example 512x512 image and 3x3 kernel
    image = imread("../Materials/lena_gray.png", "L").astype('float32')
    polarimage = cart2pol_2d(image, theta_res=2*np.pi/1024.)[0]
    polarshape = polarimage.shape

    # Kernel
    kern = np.zeros([3,3], dtype=np.complex)
    kern.real = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    # kern.imag = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    # kern.real = np.random.random([3,3])
    padded_kern = np.pad(kern,
                         [[polarshape[0]/2-1, polarshape[0]/2-2],
                          [polarshape[1]/2-1,polarshape[1]/2-2]],
                         constant_values=0,mode='constant')

    # Fourier transform
    fftpolar = fftshift(fft2(polarimage))
    fftpolar = convolve2d(fftpolar, kern.real) # Done in shifted domain, i.e. origin at center
    ifftout = ifft2(fftshift(fftpolar))

    fftkern = fft2(padded_kern)
    fftout = polarimage * np.abs(fftkern)


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(np.abs(pol2cart_2d(ifftout)[0]))
    ax2.imshow(np.abs(pol2cart_2d(fftout)[0]))
    plt.show()
    pass

if __name__ == '__main__':
    main()
