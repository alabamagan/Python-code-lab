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

        im = interp.interp2d(x, y, image, fill_value=0, bounds_error=False, kind='linear')

        T, THETA = np.meshgrid(t, thetas)
        T = T+1E-8
        THETA = THETA % (2*np.pi)

        pool = Pool(processes=10)

        z_upper = np.ones(T.shape)*T.max()
        z_lower = np.ones(T.shape)*T.min()
        out = np.zeros(T.flatten().shape)
        pts = zip(T.flatten(), THETA.flatten(), z_upper.flatten(), z_lower.flatten())

        arguments = [[im, p] for p in pts]
        for i, x in enumerate(pool.map(FilteredProjection._forwardIntergrate, arguments)):
            out[i] = x

        out = out.reshape(T.shape)
        return out

    @staticmethod
    def backward(sino, thetas, FOV=None):
        assert isinstance(sino, np.ndarray)
        assert isinstance(thetas, np.ndarray)
        assert thetas.ndim == 1
        assert sino.ndim == 2

        # Assume y axis is theta
        fftsino = fftshift(fft(sino, axis=1), 1)

        if FOV is None:
            FOV = np.int(np.sqrt(sino.shape[0]**2/2.)/2.)
        out = np.zeros([FOV*2, FOV*2])
        x = np.arange(FOV*2)-FOV/2
        y = np.arange(FOV*2)-FOV/2
        x, y = np.meshgrid(x, y)

        # Filtering
        f = np.linspace(-sino.shape[1]/2, sino.shape[1]/2-1, sino.shape[1])
        f = np.abs(f).astype(np.complex)
        f.imag = f.real
        f = np.stack([f for i in xrange(sino.shape[0])], axis=0)

        fftsino = np.multiply(f,fftsino)

        # G_theta(t)
        ifftsino_r = ifft(fftshift(fftsino, 1), axis=1)
        ifftsino = interp.interp2d(np.linspace(-sino.shape[1]/2, sino.shape[1]/2-1, sino.shape[1]),
                                   thetas,  ifftsino_r.real)

        pts = zip(x.flatten(), y.flatten())
        arguments = [[ifftsino, thetas]+list(p) for p in pts]

        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(ifftsino_r.real)
        # plt.show()
        #
        # for i, p in enumerate(arguments[::5]):
        #     interpobject, the, X, Y = p
        #     XX = X*np.cos(the) + Y * np.sin(the) + sino.shape[1]/2
        #     scat = ax.scatter(XX, the*512/2./np.pi, c='b')
        #     plt.draw()
        #     plt.pause(0.01)
        #     scat.remove()

        pool = Pool(10)
        # Intergration
        for i, x in enumerate(pool.map(FilteredProjection._backwardIntergrate, arguments)):
            # row major
            X = i / out.shape[1]
            Y = i % out.shape[1]
            out[X, Y] = x

        return out

    @staticmethod
    def _forwardIntergrate(a):
        # from scipy.integrate import quad
        # import numpy as np
        interpobject, pt = a
        t_i, theta_i, z_u, z_l = pt

        zs = np.linspace(z_l, z_u, int(np.ceil(z_u-z_l)))
        l = lambda z: interpobject(z*np.sin(theta_i) + t_i*np.cos(theta_i), -z*np.cos(theta_i) + t_i*np.sin(theta_i))
        return np.sum([l(z) for z in zs])

    @staticmethod
    def _backwardIntergrate(a):
        interpobject, thetas, X, Y,  = a
        # I(x, y) = sum_\theta G(x cos \theta + y sin \theta, \theta)
        l = lambda the: interpobject(X * np.cos(the) + Y * np.sin(the), the)
        return np.sum([l(t)/2/np.pi for t in thetas])/len(thetas)



def TestForward():
    import matplotlib.pyplot as plt
    from imageio import imread
    # gt = imread("../Materials/gt.tif")
    gt = np.ones([256, 256])
    gt[120:150, 200:240]=50
    gt[35:140, 35:140]=55

    # Create square phantom
    N=gt.shape[0]
    samplerange = np.int(np.sqrt(2*(N/2)**2))
    radon = FilteredProjection.forward(gt,
                                       np.linspace(0, 2*np.pi, 2*N),
                                       np.linspace(-samplerange+0.5, samplerange-0.5, int(np.ceil(samplerange*2))))

    np.save("../Materials/radon.np", radon)

def TestBackward():
    sino = np.load("../Materials/radon.np.npy")

    thetas = np.linspace(0, 2*np.pi, 512)
    out = FilteredProjection.backward(sino, thetas)

    plt.imshow(out)
    plt.show()

if __name__ == '__main__':
    TestBackward()