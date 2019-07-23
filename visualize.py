import matplotlib.pyplot as plt

def visualize(X, fname, shape):
    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    for i in range(0, 100):
        plt.subplot(10, 10, i + 1)
        plt.tick_params(labelleft='off', top='off', bottom='off')
        plt.tick_params(labelbottom='off', left='off', right='off')
        plt.imshow(X[i].reshape(shape), cmap='gray')
    plt.savefig(fname)
    plt.close()

def visualize_kernel(W, fname):
    plt.figure(num=None, figsize=(22, 22), dpi=60, facecolor='w', edgecolor='k')
    for y in range(22):
        for x in range(22):
            plt.subplot(22, 22, x + y * 22 + 1)
            plt.tick_params(labelleft='off', top='off', bottom='off')
            plt.tick_params(labelbottom='off', left='off', right='off')
            plt.imshow(W.reshape((1, 22, 22, 1, 28, 28))[0, y, x, 0, y:y+11, x:x+11], cmap='gray')
    plt.savefig(fname)
    plt.close()
    