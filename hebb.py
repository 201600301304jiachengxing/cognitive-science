import numpy as np

def draw_bin_image(image_matrix):
    for row in image_matrix.tolist():
        print('| ' + ' '.join(' *'[int((val+1)/2)] for val in row))
    print('\n')

def acti_fun(x):
    for i in range(x.size):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = -1
    return x

class Hebb(object):
    def __init__(self, n):
        self.n = n
        self.weights = np.zeros((self.n, self.n))

    def wild_hebb(self, vec):
        raw = self.weights * vec.repeat(self.n).reshape(self.n, self.n)
        new = np.zeros_like(raw.reshape(30, 30))
        u, v = new.shape
        for i in range(u):
            for j in range(v):
                new[i, j] = np.sum(raw[0:j+1, i])
        delta_w = vec.repeat(self.n).reshape(self.n, self.n) * new
        return delta_w


    def train(self, input_vector, iter, rate, use_GEN=False, use_norm=False, alaph = 1e-3):
        for i in range(iter):
            for vec in input_vector:
                vec = np.matrix(vec)
                delta = np.zeros((self.n, self.n))
		#这里可以使用广义hebb算法中的权重更新算法
                if use_GEN:
                    wild_hebb = self.wild_hebb(vec)
                    delta = np.ones_like(wild_hebb)
                    index = np.where(wild_hebb < 0)
                    delta[index] = -1
                if use_norm:
                    self.weights = self.weights + rate * (vec.getT().dot(vec) - delta - alaph * vec.repeat(self.n).reshape(self.n, self.n) * self.weights)
                else:
                    self.weights = self.weights + rate * (vec.getT().dot(vec) - delta)
            print(i)
        return self.weights

    def predict(self, input_vector):
         return self.weights.dot(np.matrix(input_vector).getT())

zero = [
    -1, 1, 1, 1, -1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    -1, 1, 1, 1, -1
    ]

one = [
    -1, 1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1
    ]

two = [
    1, -1, 1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, 1, -1,
    -1, 1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, 1, 1, 1,
    ]

half_zero = [
    1, 1, 1, 1, 1,
    1, -1, -1, -1, 1,
    1, -1, 1, -1, 1,
    1, -1, -1, -1, 1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    ]

half_one = [
    -1, 1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, 1, -1, -1, -1,
    1, -1, -1, -1, -1,
    -1, -1, -1, -1, 1
    ]

half_two = [
    -1, 1, -1, -1, 1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, -1, -1,
    -1, 1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, 1, 1, 1,
    ]


hebb  = Hebb(30)
w=hebb.train([zero, one, two], 300, 0.2, use_GEN=False, use_norm=True)

def predict(num, half_num):
    pre = hebb.predict(half_num)
    # print(pre.reshape((6,5)))
    draw_bin_image(acti_fun(np.array(num)).reshape((6, 5)))
    draw_bin_image(acti_fun(np.array(half_num)).reshape((6, 5)))
    draw_bin_image(acti_fun(pre).reshape((6, 5)))


predict(zero, half_zero)
predict(one, half_one)
predict(two, half_two)

