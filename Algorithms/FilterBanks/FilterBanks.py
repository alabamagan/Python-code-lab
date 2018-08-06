import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from abc import ABCMeta, abstractmethod

class FilterBankNodeBase(object):
    __metaclass__ = ABCMeta
    def __init__(self, inNode=None):
        self._input_node = None
        self._referencing_node = []
        self._core_matrix = None
        self._coset_vectors = []
        self.hook_input(inNode)
        pass

    def __del__(self):
        if self._input_node != None:
            assert issubclass(type(self._input_node), FilterBankNodeBase)
            if self in self._input_node._referencing_node:
                self._input_node._referencing_node.remove(self)

    @abstractmethod
    def _core_function(self, inNode):
        return inNode

    def run(self, input):
        if self._input_node is None:
            return self._core_function(input)
        else:
            return self._core_function(self._input_node.run(input))

    def hook_input(self, filter):
        if filter == None:
            return

        assert issubclass(type(filter), FilterBankNodeBase)
        self._input_node = filter

    def input(self):
        return self._input_node

    def _calculate_coset(self):
        assert not self._core_matrix is None

        M = int(np.abs(np.linalg.det(self._core_matrix)))
        inv_coremat = np.linalg.inv(self._core_matrix.T)

        # For 2D case
        for i in xrange(M):
            for j in xrange(M):
                # Costruct coset vector
                v = np.array([i,j], dtype=np.float)
                f = inv_coremat.dot(v)
                if np.all((0<=f) & (f < 1.)):
                    self._coset_vectors.append(v.astype('int'))



    @staticmethod
    def periodic_modulus_2d(arr, xrange, yrange):
        """
        Description
        -----------

        Modular the range of the input 2D array of vector to desired range [a, b].
        The input shape should by (X, Y, 2).

        f(y)    = y                 if  y \in [a, b]
                = y - a - n(b-a+1)  if  otherwise       for n \in \mathbb{Z}

        In general, [a, b] is a one of the partitions of \mathbb{Z} denoted as:
            Part(Z;[a, b]) = \{[a-n(b-a+1), b-n(b-a+1)]; n\in \mathbb{Z}\}

        When n = 0, the element range is [a, b]. The idea is to find the closest partition,
        defined by n, and translate the number to its unit cell, then do modulus.

        """
        assert isinstance(arr, np.ndarray)
        assert arr.ndim==3, "Array shape should be (X_shape, Y_shape, 2)"
        assert arr.shape[2] == 2, "Array shape should be (X_shape, Y_shape, 2)"
        assert len(xrange) == len(yrange) == 2
        assert xrange[1] > xrange[0] and yrange[1] > yrange[0]

        mx = np.invert((xrange[0] <= arr[:,:,0]).astype('bool') & (arr[:,:,0] <= xrange[1]).astype('bool'))
        my = np.invert((yrange[0] <= arr[:,:,1]).astype('bool') & (arr[:,:,1] <= yrange[1]).astype('bool'))
        arx = arr[:,:,0]
        ary = arr[:,:,1]
        rx = xrange[1] - xrange[0] + 1
        ry = yrange[1] - yrange[0] + 1
        Nx = np.floor((arx[mx] - xrange[0])/rx)
        Ny = np.floor((ary[my] - yrange[0])/ry)

        arx[mx] = arx[mx] - Nx*rx
        ary[my] = ary[my] - Ny*ry
        return arr

    @staticmethod
    def unitary_modulus_2d(arr, xrange, yrange):
        assert isinstance(arr, np.ndarray)
        assert arr.ndim==3, "Array shape should be (X_shape, Y_shape, 2)"
        assert arr.shape[2] == 2, "Array shape should be (X_shape, Y_shape, 2)"
        assert len(xrange) == len(yrange) == 2
        assert xrange[1] > xrange[0] and yrange[1] > yrange[0]

        arr[np.invert(xrange[0] <arr[:,:,0] < xrange[1])] = 0
        arr[np.invert(yrange[1] <arr[:,:,1] < yrange[1])] = 0
        return arr

class Downsample(FilterBankNodeBase):
    def __init__(self, inNode=None):
        FilterBankNodeBase.__init__(self, inNode)
        self._core_matrix = np.array([[1, 1],
                                      [1,  -1]])
        self._calculate_coset()

    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.ndim == 2
        assert inflow.shape[0] == inflow.shape[1]

        # if not complex, assume x-space input, do fourier transform
        if not np.any(np.iscomplex(inflow)):
            self._inflow = np.fft.fftshift(np.fft.fft2(inflow))
        else:
            self._inflow = np.copy(inflow)

        s = inflow.shape[0]

        u, v = np.meshgrid(np.arange(s) - s//2, np.arange(s)-s//2)
        omega = np.stack([u, v], axis=-1)

        # Number of bands for the given core matrix to achieve critical sampling.
        omega = [(omega - 2*np.pi*v).dot(np.linalg.inv(self._core_matrix.T)) for v in self._coset_vectors]

        # Periodic modulus
        omega = [FilterBankNodeBase.periodic_modulus_2d(o, [-s//2, s//2-1], [-s//2, s//2-1]) for o in omega]
        outflow = np.zeros(self._inflow.shape, dtype=self._inflow.dtype)

        for i in xrange(outflow.shape[0]):
            for j in xrange(outflow.shape[1]):
                for o in omega:
                    if o[i,j, 0] % 1 == 0 and o[i,j,1] % 1 == 0:
                        outflow[i,j] += self._inflow[int(o[i,j,0] + s//2),
                                                     int(o[i,j,1] + s//2)] \
                                        / float(len(self._coset_vectors))

        return outflow


class Upsample(FilterBankNodeBase):
    def __init__(self, coset_vector, inNode=None):
        FilterBankNodeBase.__init__(self, inNode)

        self._coset_vector = coset_vector
        self._core_matrix = np.array([[1, 1],
                                      [1,  -1]])
        self._calculate_coset()

    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.ndim == 2
        assert inflow.shape[0] == inflow.shape[1]




if __name__ == '__main__':
    from imageio import imread
    import matplotlib.pyplot as plt

    im = imread('../../Materials/lena_gray.png')[:,:,0]

    # H_0 = Downsample(None)
    # H_0 = Downsample(np.array([-0.5, 0]))
    H_0 = Downsample()
    # out = H_0._core_funct/ion(np.fft.fftshift(np.fft.fft2(im)))
    out = H_0.run(im)

    # print out.shape
    # plt.scatter(out[:,:,0].flatten(), out[:,:,1].flatten())
    # plt.imshow(out[:,:,0])
    plt.imshow(np.fft.ifft2(np.fft.fftshift(out)).real)           # Some times need fftshift, sometimes doesn't
    # plt.imshow(np.fft.ifft2(out).real)
    # plt.imshow(np.abs(out))
    plt.show()
