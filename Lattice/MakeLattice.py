import numpy as np
import matplotlib.pyplot as plt

from Algorithms.FilterBanks.FilterBanks import Downsample, Upsample

class Lattice2D(object):
    def __init__(self):
        super(Lattice2D, self).__init__()
        self._basis = None

    def sample_lattice(self, repeat):
        assert not self._basis is None
        assert isinstance(self._basis, np.ndarray)

        lattice = [self._basis[0], self._basis[1]]
        for a in xrange(-repeat, repeat+1):
            for b in xrange(-repeat, repeat+1):
                lattice.append(a*self._basis[0] + b*self._basis[1])

        return np.array(lattice)


def plotTransformQuiver(d, F=5):
    fig, ax = plt.subplots(1, 1)
    X = np.array(zip(d._uv[:,:,0].flatten(), d._uv[:,:,1].flatten()))
    XYX = np.array(zip(d._omega[0][:,:,0].flatten(), d._omega[0][:,:,1].flatten()))
    XYX2 = np.array(zip(d._omega[1][:,:,0].flatten(), d._omega[1][:,:,1].flatten()))

    ax.scatter(X[:,0], X[:,1])
    ax.scatter(XYX2[:,0], XYX2[:,1], s=5)
    ax.quiver(d._uv[:,:,0].flatten()[::F], d._uv[:,:,1].flatten()[::F] + d._uv[:,:,0].flatten()[::F] * 0.05,
              d._omega[1][:,:,0].flatten()[::F] - d._uv[:,:,0].flatten()[::F],
              d._omega[1][:,:,1].flatten()[::F] - d._uv[:,:,1].flatten()[::F],
              # d._omega[1][:,:,0].flatten()[::F] - d._omega[0][:,:,0].flatten()[::F],
              # d._omega[1][:,:,1].flatten()[::F] - d._omega[0][:,:,1].flatten()[::F],
              scale=1, scale_units='xy', alpha=0.5, width=0.002)
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.axis('equal')
    plt.show()


def plotScatter(d):

    fig, ax = plt.subplots(1, 1)
    X = np.array(zip(d._uv[:,:,0].flatten(), d._uv[:,:,1].flatten()))
    XYX = np.array(zip(d._omega[0][:,:,0].flatten(), d._omega[0][:,:,1].flatten()))
    XYX2 = np.array(zip(d._omega[1][:,:,0].flatten() + d._coset_vectors[1][0],
                        d._omega[1][:,:,1].flatten() + d._coset_vectors[1][1]))

    ax.scatter(X[:,0], X[:,1])
    ax.scatter(XYX[:,0], XYX[:,1], s=5)
    ax.scatter(XYX2[:,0], XYX2[:,1], s=5, c='r')
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    B1 = np.array([[1,0],[0,1]])
    D0 = np.array([[2, 0], [0, 1]])
    # D0 = np.linalg.inv(D0).T
    B2 = B1.dot(D0)

    test = Lattice2D()
    test._basis = np.array(B1)
    lattice1 = test.sample_lattice(3)
    test._basis = np.array(B2)
    lattice2 = test.sample_lattice(3)

    d = Upsample()
    d.set_core_matrix(np.array([[2, 0], [0, 1]]))
    out = d.run(np.random.random([16,16, 2]))


    plotTransformQuiver(d, 1)
    # plotScatter(d)





