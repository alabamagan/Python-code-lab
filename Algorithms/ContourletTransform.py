import numpy as np
from GeometricMasks import ring_mask, pie_section_mask
from numpy.fft import ifft2, fft2, fftshift, ifftshift

class PolarContourletTransform(object):
    def __init__(self, level=4):
        self.level=level
        self.pmasks = None
        self.rmasks = None

    def _forwardTransform(self, input):
        assert isinstance(input, np.ndarray), "Input must be np array!"

        s = list(input.shape)

        # Create masks
        rad = [0] +  [min(s) / 2.**i for i in xrange(self.level - 1)][::-1] + [1E5]
        assert rad.count(0) == 1, "Reduce number of level."
        angs = np.linspace(0, 180, 2**(self.level-1) + 1)[:-1]
        width = 180 / (2.**(self.level-1)-1)
        self.pmasks = [np.invert(pie_section_mask(s,
                                                  np.deg2rad(angs[i] + width/2.),
                                                  np.deg2rad(angs[i] - width/2.)))
                  for i in xrange(len(angs))]
        self.rmasks = [np.invert(ring_mask(s,rad[i+1],rad[i])) for i in xrange(len(rad)-1)]

        # Apply masks, grid define by level
        outdict = {}
        fftim = fftshift(fft2(input))
        for i, m in enumerate(self.rmasks):
            fftim_copy = np.copy(fftim)
            fftim_copy[m] = np.complex(0)
            inner_outdict = {}
            for j, p in enumerate(self.pmasks):
                fftim_pmask_copy = np.copy(fftim_copy)
                fftim_pmask_copy[p] = np.complex(0)
                inner_outdict[j] = ifft2(fftshift(fftim_pmask_copy))
                # inner_outdict[j] = np.invert(p) * np.invert(m)
            outdict[i] = inner_outdict

        return outdict



    def _backwardsTransform(self, input):
        assert isinstance(input, np.ndarray), "Input must be np array!"


if __name__ == '__main__':
    from imageio import imread
    import matplotlib.pyplot as plt

    gt = imread("../Materials/gt.tif")
    im = imread("../Materials/s0.tif")

    transformer = PolarContourletTransform(4)
    a1 = transformer._forwardTransform(gt)
    a2 = transformer._forwardTransform(im)

    fig, figarr = plt.subplots(len(a1.keys())*2, len(a1[0].keys()))
    for ki in a1:
        vali = a1[ki]
        for kj in vali:
            valj = vali[kj]
            figarr[2*ki, kj].imshow(np.real(valj), cmap='jet', vmin=-200./(ki+1), vmax=200./(ki+1))
            figarr[2*ki, kj].axis('off')
            figarr[2*ki + 1, kj].imshow(np.real(a2[ki][kj]), cmap='jet', vmin=-200./(ki+1), vmax=200./(ki+1))
            figarr[2*ki + 1, kj].axis('off')
    mnt = plt.get_current_fig_manager()
    mnt.window.showMaximized()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.02, top=0.98, wspace=0.05, hspace=0.05)
    plt.show()

    # a2 = transformer._forwardTransform(im)