import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class ReconstractionTICA(chainer.Chain):

    def __init__(self):
        super(ReconstractionTICA, self).__init__()
        with self.init_scope():
            self.f = L.Linear(1 * 28 * 28, 1 * 22 * 22)
            self.p = F.average_pooling_2d
            self.g = L.Linear(1 * 22 * 22, 1 * 28 * 28)
            self.mask = np.zeros((1, 22, 22, 1, 28, 28), dtype=float)
            for dy in range(22):
                for dx in range(22):
                    self.mask[0, dy, dx, 0, dy:dy+11, dx:dx+11] = 1.0
            self.mask = self.mask.reshape((1 * 22 * 22, 1 * 28 * 28))

    def __call__(self, x):
        self.f.W.data = self.f.W.data * self.mask
        self.g.W.data = self.g.W.data * self.mask.T
        h = self.f(x)
        h1 = F.square(h)
        h1 = F.reshape(h1, (-1, 1, 22, 22))
        h1 = self.p(h1, 5, stride=1, pad=0)
        h1 = F.sqrt(h1 + 0.0001)
        h2 = self.g(h)
        return (h1, h2)

