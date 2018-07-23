from scipy.interpolate import RegularGridInterpolator
from scipy.misc import imsave
import numpy as np


def pol2cart(r, theta, cent=None):
    # default center is [0,0]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if not cent is None:
        assert len(cent) == 2, "Shape of center is incorrect"
        x += cent[0]
        y += cent[1]
    return x ,y

def cart2pol(x, y, cent=None):
    if not cent is None:
        assert len(cent) == 2, "Shape of center is incorrect"
        tx = x - cent[0]
        ty = y - cent[1]
    else:
        tx = np.copy(x)
        ty = np.copy(y)
    tx = tx.astype('float')
    ty = ty.astype('float')

    r = np.sqrt(tx**2 + ty**2)
    theta =np.arctan2(ty, tx) % (2 * np.pi)
    return r, theta

def cart2pol_2d(im, cent=None, r_res=1., theta_res=2*np.pi/512., theta_range=(0,2*np.pi)):
    """
    Description
    -----------
      Resample an image into polar coordinate.

    :param np.ndarray im: Input image in cartesian space
    :param tuple cent: Define center of polar coordinate conversion
    :param float r_res: Resolution of output image in radial direction
    :param float theta_res: Resolution of output image in angular direction
    :param tuple theta_range: Range of theta to interpolate
    :return: (polar, (r, theta))
    """

    # Copy input to new variables
    imageCopy = np.copy(im).astype('float')
    imageShape = imageCopy.shape

    # Define center of polar conversion
    if cent is None:
        cent = (np.array(imageShape) - 1)/ 2.

    # Determine output image properties
    sampling_radius = np.sqrt((imageShape[0] - cent[0]) ** 2 + (imageShape[1]- cent[1]) **2)

    # Define output image polar coordinate grid
    r = np.linspace(0, sampling_radius, int(sampling_radius / float(r_res)))
    theta = np.linspace(theta_range[0],
                        theta_range[1],
                        1 + int(abs(theta_range[1] - theta_range[0])/float(theta_res)))[:-1]
    r, theta = np.meshgrid(r, theta)

    # Define corresponding sampling grid
    rx, ry = pol2cart(r, theta, cent)
    coords = np.array(zip(rx.flatten(), ry.flatten()))

    # Resample
    interp_image = RegularGridInterpolator([np.arange(imageShape[0]), np.arange(imageShape[1])],
                                           imageCopy,
                                           fill_value=0,
                                           bounds_error=False)
    out_image = interp_image(coords).reshape(rx.shape)
    return out_image, (r, theta)

def pol2cart_2d(im, cent=None, r=None, theta=None, w_res=1., h_res=1.):
    imageCopy = np.copy(im)
    imageShape = imageCopy.shape

    # Default input grid definition
    if r is None:
        r = np.linspace(0, imageShape[1] - 1, imageShape[1])

    if theta is None:
        theta = np.linspace(0, np.pi * 2, imageShape[0] + 1)[:-1]

    sampling_size = int(np.sqrt((r[-1]-1)**2/2.))*2

    # Define samplilng grid
    x, y = np.meshgrid(np.arange(sampling_size), np.arange(sampling_size))
    x, y = [k.astype('float') for k in [x, y]]

    if cent is None:
        cent = [x.max() / 2., y.max() / 2.]

    xr, xtheta = cart2pol(x, y, cent)
    xtheta = xtheta % (2*np.pi)
    coords = np.array(zip(xtheta.flatten(), xr.flatten()))

    interp_polar = RegularGridInterpolator([theta, r], imageCopy, fill_value=None, bounds_error=False)
    out_image = interp_polar(coords).reshape(xr.shape).T # we use x, y convention, but its actually y, x
    return out_image, (x, y)


if __name__ == '__main__':
    from scipy.ndimage import imread
    import matplotlib.pyplot as plt

    image = imread("../Materials/lena_gray.png", 'L')
    polar, pcoords = cart2pol_2d(image, theta_res=2*np.pi / 2048)
    image2 = pol2cart_2d(polar, r=pcoords[0][0], theta=pcoords[1][:, 0])[0]

    fig = plt.figure()
    ax1, ax2 = [fig.add_subplot(i) for i in [121, 122]]
    ax1.imshow(polar)
    ax2.imshow(image2)
    plt.show()




