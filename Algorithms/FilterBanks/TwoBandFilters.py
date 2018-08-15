from FilterBanks import Upsample, Downsample, FilterBankNodeBase
from abc import ABCMeta, abstractmethod
import numpy as np

class TwoBandDownsample(Downsample):
    def __init__(self, inNode=None):
        super(TwoBandDownsample, self).__init__(inNode)

    def _core_function(self, inflow):
        r"""

        :param inflow:
        :return:
        """

        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.shape[0] == inflow.shape[1]

        if inflow.ndim == 2:
            self._outflow = super(TwoBandDownsample, self)._core_function(inflow)
            return self._outflow
        else:
            self._outflow = np.concatenate([super(TwoBandDownsample, self)._core_function(inflow[:, :, i])
                                            for i in xrange(inflow.shape[-1])], axis=2)
            return self._outflow


''' Testing'''
if __name__ == '__main__':
    from imageio import imread
    import matplotlib.pyplot as plt
    from numpy.fft import fftshift, fft2, ifft2, ifftshift

    im = imread('../../Materials/lena_gray.png')[:,:,0]
    # These phase shift operations can produce fan filter effect, but introduce the need to do fftshift
    x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    px = np.zeros(im.shape, dtype=np.complex)
    px.imag = -np.pi*x
    px = np.exp(px)
    im = im*px

    # shift the input so that origin is at center of image
    s_fftim = fftshift(fft2(ifftshift(im)))
    im = imread('../../Materials/lena_gray.png')[:,:,0]
    H_0 = TwoBandDownsample()
    # H_0.set_shift([0.5, 0])
    H_1 = TwoBandDownsample()
    # H_1.set_shift([0.5, 0])
    H_1.hook_input(H_0)

    out = H_1.run(s_fftim)

    '''Display subband components'''
    M = out.shape[-1]
    ncol=2
    fig, axs = plt.subplots(M / ncol, ncol)
    for i in xrange(M):
        if M <= ncol:
            try:
                axs[i].imshow(ifft2(fftshift(H_1.get_subband(i))).real)
                # axs[i].imshow(np.abs(H_1.get_subband(i)), vmin=0, vmax=2500)
            except:
                pass
            # axs[i].imshow(ifft2(H_0._outflow[:,:,i]).real)
        else:
            axs[i//ncol, i%ncol].imshow(ifft2(fftshift(H_1.get_subband(i))).real)
            # axs[i//ncol, i%ncol].imshow(np.abs(H_1.get_subband(i)), vmin=0, vmax=2500)
    plt.show()