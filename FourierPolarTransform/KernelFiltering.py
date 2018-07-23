import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt



def main():
    """
    This example shows how a 3x3 image filter kernel operates in the fourier space formally.

    :return:
    """

    # Generate an example 512x512 image and 3x3 kernel
    image = ndimage.imread("../Materials/lena_gray.png", "L").astype('float32')

    # Kernel
    kern = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    # Fourier transform
    fftkern = np.fft.fftshift(np.fft.fft2(np.pad(kern,
                                                 [[256-1, 256-2], [256-1,256-2]],
                                                 constant_values=0,
                                                 mode='constant')
                                          ))
    fftimage = np.fft.fftshift(np.fft.fft2(image))

    # Multiplication in fourier space
    fftoutput = fftkern*fftimage

    # Inverse fourier transform
    ioutput = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fftoutput)))
    ioutput = np.real(ioutput)

    # Scipy output as ground truth
    spoutput = ndimage.filters.convolve(image, kern)

    fig = plt.figure()
    ax1, ax2 = [fig.add_subplot(i) for i in [211, 212]]
    ax1.imshow(np.real(ioutput), cmap="Greys_r")
    ax2.imshow(spoutput, cmap="Greys_r")
    plt.show()

if __name__ == '__main__':
    main()