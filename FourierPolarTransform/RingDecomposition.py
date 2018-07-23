from imageio import imread
from numpy.fft import fft2, ifft2, fftshift
import numpy as np

from Algorithms.GeometricMasks import ring_mask
import matplotlib.pyplot as plt


def main():
    gt = imread("../Materials/gt.tif")
    im = imread("../Materials/s0.tif")

    # Fourier transform
    fftgt = fftshift(fft2(gt))
    fftim = fftshift(fft2(im))


    maskradius = [0] + [256 / 2.**i for i in xrange(4)][::-1] + [1E5]
    fgt = [masking(fftgt, maskradius[i], maskradius[i+1]) for i in xrange(len(maskradius) - 1)]
    fim = [masking(fftim, maskradius[i], maskradius[i+1]) for i in xrange(len(maskradius) - 1)]

    fig, subplots= plt.subplots(3, len(fgt) + 1)
    for i in xrange(len(fgt)):
        subplots[0][i].imshow(np.real(ifft2(fftshift(fgt[i][0]))), cmap="Greys_r", vmin=-2E2, vmax=2E2)
        subplots[1][i].imshow(np.real(ifft2(fftshift(fim[i][0]))), cmap="Greys_r", vmin=-2E2, vmax=2E2)
        subplots[2][i].imshow(np.abs(fim[i][0]), cmap="Greys_r", vmin=0, vmax=1E5)
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')

    # Merge all back together
    sumFgt = np.stack(fgt)[:,0].sum(axis=0)
    sumFim = np.stack(fim)[:,0].sum(axis=0)
    subplots[0][len(fgt)].imshow(np.real(ifft2(fftshift(sumFgt))))
    subplots[1][len(fgt)].imshow(np.real(ifft2(fftshift(sumFim))))
    subplots[2][len(fgt)].imshow(np.abs(sumFim), vmin=0, vmax=1E5)
    subplots[0][len(fgt)].axis('off')
    subplots[1][len(fgt)].axis('off')
    subplots[2][len(fgt)].axis('off')

    plt.show()

def masking(im, start_radius, end_radius):
    assert isinstance(im, np.ndarray)

    outim = np.copy(im)
    piemask = ring_mask(outim, end_radius, start_radius)
    outim[np.invert(piemask)] = np.complex(0,0)
    return outim, piemask

if __name__ == '__main__':
    main()
    # pie_section_mask([512,512], np.deg2rad(30), np.deg2rad(15))
