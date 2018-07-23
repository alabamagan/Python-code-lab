from imageio import imread
from numpy.fft import fft2, ifft2, fftshift
import numpy as np

from Algorithms.PolarTransform import pol2cart_2d, cart2pol_2d
from Algorithms.GeometricMasks import pie_section_mask
import matplotlib.pyplot as plt


def main():
    gt = imread("../Materials/gt.tif")
    im = imread("../Materials/s0.tif")

    # Fourier transform
    fftgt = fftshift(fft2(gt))
    fftim = fftshift(fft2(im))

    # Masking
    # ringmask = ring_mask(fftgt, 160, 80)
    # fftim[np.invert(ringmask)] = np.complex(0, 0)
    # fftgt[np.invert(ringmask)] = np.complex(0, 0)
    maskangles = np.linspace(0, 180, 9)[:-1]
    maskangles = np.deg2rad(maskangles)
    fgt = [masking(fftgt, i, np.deg2rad(180/8.)) for i in maskangles]
    fim = [masking(fftim, i, np.deg2rad(180/8.)) for i in maskangles]


    ifftim = ifft2(fftshift(fftim))
    fft_ifft_im = fft2(ifftim)

    # Polar Transform
    # polfftgt = cart2pol_2d(fftgt.real)[0].astype('complex')
    # polfftgt.imag = cart2pol_2d(fftgt.imag)[0]
    # polfftim = cart2pol_2d(fftim.real)[0].astype('complex')
    # polfftim.imag = cart2pol_2d(fftim.imag)[0]

    fig, subplots= plt.subplots(3, len(maskangles) + 1)
    for i in xrange(len(maskangles)):
        subplots[0][i].imshow(np.real(ifft2(fftshift(fgt[i][0]))), cmap="jet", vmin=-2E2, vmax=2E2)
        subplots[1][i].imshow(np.real(ifft2(fftshift(fim[i][0]))), cmap="jet", vmin=-2E2, vmax=2E2)
        subplots[2][i].imshow(np.abs(fim[i][0]), cmap="jet", vmin=0, vmax=1E5)
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')

    # Merge all back together
    sumFgt = np.stack(fgt)[:,0].sum(axis=0)
    sumFim = np.stack(fim)[:,0].sum(axis=0)
    subplots[0][len(maskangles + 1)].imshow(np.real(ifft2(fftshift(sumFgt))))
    subplots[1][len(maskangles + 1)].imshow(np.real(ifft2(fftshift(sumFim))))
    subplots[2][len(maskangles + 1)].imshow(np.abs(sumFim), vmin=0, vmax=1E5)
    subplots[0][len(maskangles + 1)].axis('off')
    subplots[1][len(maskangles + 1)].axis('off')
    subplots[2][len(maskangles + 1)].axis('off')

    plt.show()

def masking(im, angle, width):
    assert isinstance(im, np.ndarray)

    outim = np.copy(im)
    piemask = pie_section_mask(outim, angle + width/2., angle - width/2.)
    outim[np.invert(piemask)] = np.complex(0,0)
    return outim, piemask

if __name__ == '__main__':
    main()
    # pie_section_mask([512,512], np.deg2rad(30), np.deg2rad(15))
