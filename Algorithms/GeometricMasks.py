import numpy as np

def ring_mask(im, radius_max, radius_min=0):
    assert isinstance(im, np.ndarray) or isinstance(im, list), "First argument should be " \
                                                            "list or a numpy array!"
    assert radius_max > radius_min

    if isinstance(im, np.ndarray):
        s = im.shape
    else:
        assert len(im) == 2
        s = list(im)

    x, y = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
    x = x.astype('float')
    y = y.astype('float')

    cent = np.array([(s[0]-1) / 2., (s[1]-1)/2.])
    mask1 = (x - cent[0])**2 + (y-cent[1])**2 <= radius_max**2.
    mask2 = (x-cent[0])**2 + (y-cent[1])**2 >= radius_min**2.
    mask = mask1 & mask2

    return mask

def pie_section_mask(im, end_theta, start_theta):
    assert isinstance(im, np.ndarray) or isinstance(im, list), "First argument should be " \
                                                            "list or a numpy array!"
    assert end_theta > start_theta
    # assert 0 < end_theta <= np.pi and 0 <= start_theta < np.pi

    if isinstance(im, np.ndarray):
        s = im.shape
    else:
        assert len(im) == 2
        s = list(im)

    # Define pie-section under

    x, y = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
    x = x.astype('float')
    y = y.astype('float')
    cent = np.array([(s[0]-1) / 2., (s[1]-1)/2.])

    theta = np.arctan2(y-cent[0], x-cent[1]) % np.pi

    mask = ((start_theta <= theta)  & (theta <= end_theta)) | \
           ((start_theta - np.pi  <= theta) & (theta <= end_theta - np.pi)) | \
           ((start_theta + np.pi  <= theta) & (theta <= end_theta + np.pi))
    return mask