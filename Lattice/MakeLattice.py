import numpy as np
import matplotlib.pyplot as plt

from Algorithms.FilterBanks.FilterBanks import Downsample

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

    d = Downsample()
    out = d.run(np.random.random([16,16]))


    # x, y = np.meshgrid(np.arange(5) - 2, np.arange(5) - 2)
    # XY = np.stack([x, y])
    # angle = np.deg2rad(30)
    # M = np.array([[np.cos(angle), -np.sin(angle)],
    #               [np.sin(angle), np.cos(angle)]])
    # XY = M.dot(XY)
    # X = np.array(zip(x.flatten(), y.flatten()))
    # XYX = np.array(zip(XY[:,:,0].flatten(), XY[:,:,1].flatten()))

    X = np.array(zip(d._uv[:,:,0].flatten(), d._uv[:,:,1].flatten()))
    XYX = np.array(zip(d._omega[0][:,:,0].flatten(), d._omega[0][:,:,1].flatten()))

    fig, ax = plt.subplots(1, 1)
    # ax.scatter(lattice1[:,0], lattice1[:,1])
    # ax.scatter(lattice2[:,0]+1, lattice2[:,1], s=4)
    ax.scatter(X[:,0], X[:,1])
    ax.scatter(XYX[:,0], XYX[:,1], s=5)
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





