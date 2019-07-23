import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import numpy as np

from model import ReconstractionTICA
from visualize import visualize, visualize_kernel

# Define constants
l = 1.0  # Reconstruction coefficient
N = 100  # Minibatch size
SNAPSHOT_INTERVAL = 10


def main():
    train, test = chainer.datasets.get_mnist(ndim=3, withlabel=True)

    nn = ReconstractionTICA()

    # Setup optimizers
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(nn)

    # (Validate input images)
    to_be_show = np.array([t[0] for t in test[:100]])
    visualize(to_be_show, 'validation.png', (28, 28))

    train = np.array([t[0] for t in train])

    # Training
    for epoch in range(200):

        # (Validate generated images)
        if (epoch % SNAPSHOT_INTERVAL == 0):
            y, z = nn(chainer.cuda.to_cpu(to_be_show))
            visualize(chainer.cuda.to_cpu(z.data), 'z.png', (28, 28))
            visualize(chainer.cuda.to_cpu(y.data), 'y.png', (18, 18))
            visualize_kernel(chainer.cuda.to_cpu(nn.f.W.data), 'W.png')

        # (Random shuffle samples)
        train = np.random.permutation(train)

        total_loss_reg = 0.0
        total_loss_rec = 0.0

        for n in range(0, len(train), N):

            x = chainer.cuda.to_cpu(train[n:n + N].reshape((N, 1 * 28 * 28)))
            y, z = nn(x)
            # loss_reg = F.sum(y)
            # loss_rec = F.sum((x - z) ** 2)
            loss_reg = F.mean(y)
            loss_rec = F.sum((x - z) ** 2) / np.prod(x.shape)
            loss = loss_reg + l * loss_rec
            nn.cleargrads()
            loss.backward()
            optimizer.update()

            total_loss_reg += loss_reg.data
            total_loss_rec += loss_rec.data

        # (View loss)
        total_loss_reg /= len(train) / N
        total_loss_rec /= len(train) / N
        print(epoch, total_loss_reg, total_loss_rec)


if __name__ == '__main__':
    main()
