import numpy as np
import scipy.interpolate as interp
from scipy.integrate import quad
from tqdm import tqdm
import skimage.transform as trans
from numpy.fft import ifft2, fft2, fftshift, ifftshift, fft, ifft, fftfreq

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

        im = interp.RegularGridInterpolator((x, y), image, fill_value=0, bounds_error=False)

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
        fftsino = fft(sino, axis=1)

        # FOV is half the image dimension
        if FOV is None:
            FOV = np.int(np.sqrt(sino.shape[0]**2/2.)/2.)
        out = np.zeros([FOV*2, FOV*2])
        x = np.arange(-FOV, FOV)
        y = np.arange(-FOV, FOV)
        x, y = np.meshgrid(x, y)

        # Filtering
        # Ramp-filter
        f = np.abs(fftfreq(sino.shape[1], 1)) * 2
        f[f==0] = 1/float(sino.shape[0]) # Recover dc offset
        f = np.stack([f for i in xrange(sino.shape[0])], axis=0)

        fftsino = np.multiply(f,fftsino)


        # G_theta(t)
        ifftsino_r = ifft(fftsino, axis=1)

        delta_theta=(thetas[1]-thetas[0])
        for i, thi in enumerate(thetas):
            t = y * np.cos(thi) + x * np.sin(thi)
            X = np.interp(t, np.linspace(-sino.shape[1]/2, sino.shape[1]/2-1, sino.shape[1]),
                          ifftsino_r[i].real, left=0, right=0)
            out += X
        return out * delta_theta / 4. # why divide 4?


    @staticmethod
    def _forwardIntergrate(a):
        interpobject, pt = a
        t_i, theta_i, z_u, z_l = pt

        zs = np.linspace(z_l, z_u, int(np.ceil(z_u-z_l)))
        delta_zs = zs[1] - zs[0]
        pts = np.array(zip(zs*np.sin(theta_i)+t_i*np.cos(theta_i), -zs*np.cos(theta_i)+t_i*np.sin(theta_i)))
        return np.sum(interpobject(pts)*delta_zs)




def TestForward():
    import matplotlib.pyplot as plt
    from skimage.transform import radon, iradon, resize
    from imageio import imread
    gt = imread("../Materials/gt.tif")
    gt = resize(gt, [256, 256], preserve_range=True)+1000
    # gt = np.ones([256, 256])
    # gt[120:150, 200:240]=50
    # gt[35:140, 35:140]=55

    # Create square phantom
    N=gt.shape[0]
    samplerange = np.int(np.sqrt(2*(N/2)**2))
    sino  = FilteredProjection.forward(gt,
                                       np.linspace(0, 2*np.pi, 1025)[:-1],
                                       np.linspace(-samplerange, samplerange-1, 2*samplerange))

    # sino2 = radon(gt, np.rad2deg(np.linspace(0, 2*np.pi, 2*N))+90).T
    # sino2 = np.fliplr(sino2)

    # fig, subplots = plt.subplots(1, 3)
    # print sino.shape, sino2.shape
    # subplots[0].imshow(sino)
    # subplots[1].imshow(sino2)
    # subplots[2].imshow(np.abs(sino-sino2)/np.abs(sino))
    # plt.show()
    np.save("../Materials/radon_2.np", sino)

def TestBackward():
    from imageio import imread
    from skimage.transform import resize, iradon
    sino = np.load("../Materials/radon_2.np.npy")

    gt = imread("../Materials/gt.tif")
    gt = resize(gt, [256, 256], preserve_range=True)+1000

    thetas = np.linspace(0, 2*np.pi, 1025)[:-1]
    out = FilteredProjection.backward(sino, thetas, 128)
    # out = iradon(sino.T, np.rad2deg(thetas))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(gt, cmap="Greys_r", vmin=0, vmax=2000)
    ax2.imshow(out, cmap="Greys_r", vmin=0, vmax=2000)
    ax3.imshow(iradon(sino.T, np.rad2deg(thetas)).T, cmap="Greys_r", vmin=0, vmax=2000)
    plt.show()

if __name__ == '__main__':
    # TestForward()
    TestBackward()