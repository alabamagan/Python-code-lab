import numpy as np
import scipy.interpolate as interp
from scipy.integrate import quad
from tqdm import tqdm
import skimage.transform as trans
from numpy.fft import ifft2, fft2, fftshift, ifftshift, fft, ifft

from pathos.multiprocessing import ProcessingPool as Pool


import matplotlib.pyplot as plt

class FilteredProjection(object):
    def __init__(self):
        pass

    @staticmethod
    def forward(image, thetas, t):
        assert isinstance(image, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert isinstance(thetas, np.ndarray)
        assert thetas.ndim == t.ndim == 1
        assert image.ndim == 2

        # Relax this
        assert image.shape[0] == image.shape[1]

        imageshape = image.shape
        x, y = (np.arange(imageshape[0]), np.arange(image.shape[1]))
        N = imageshape[0]
        x -= N//2
        y -= N//2

        im = interp.interp2d(x, y, image, fill_value=0, bounds_error=False)

        T, THETA = np.meshgrid(t, thetas)
        T = T+1E-8
        THETA = THETA % (2*np.pi)

        pool = Pool(processes=3)

        z_upper = np.ones(T.shape)*np.sqrt(2*(N/2.)**2)
        z_lower = -np.ones(T.shape)*np.sqrt(2*(N/2.)**2)
        out = np.zeros(T.flatten().shape)
        pts = zip(T.flatten(), THETA.flatten(), z_upper.flatten(), z_lower.flatten())

        arguments = [[im, p] for p in pts]
        #
        for i, x in enumerate(pool.map(FilteredProjection._forwardIntergrate, arguments)):
            out[i] = x

        out = out.reshape(T.shape)
        return out

    @staticmethod
    def _forwardIntergrate(a):
        from scipy.integrate import quad
        import numpy as np
        interpobject, pt = a
        t_i, theta_i, z_u, z_l = pt

        l = lambda z: interpobject(z*np.sin(theta_i) + t_i*np.cos(theta_i), -z*np.cos(theta_i) + t_i*np.sin(theta_i))
        return quad(l, z_l, z_u, limit=int(z_u-z_l))[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create square phantom
    N=128
    im = np.zeros([N,N])
    im[25:50, 25:50] = 45
    im[50:100, 50:100] = 200
    radon = FilteredProjection.forward(im,
                                       np.linspace(0, 2*np.pi, 2*N),
                                       np.linspace(-N/2, N/2-1, N))
    print radon.shape

    plt.imshow(radon)
    plt.show()