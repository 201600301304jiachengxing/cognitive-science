import numpy as np

def sigmoid(x):  # 激活函数
    return 1 / (1 + np.exp(-x))

def bp(w1, w2):
    input = np.array([[0.35], [0.9]])  # 输入数据

    real = np.array([[0.5]])  # 真实值
    for s in range(0, 100, 1):
        pq = sigmoid(np.dot(w1, input))  # 第一层输出
        output = sigmoid(np.dot(w2, pq))  # 第二层输出,也即是最终输出
        e = output - real  # 误差
        if np.square(e) / 2 < 1e-6:
            print(s)
            break
        else:
            # 否则,按照梯度下降计算权重参数
            # 其中,应用链式法则计算权重参数的更新量
            w2 = w2 - e * output * (1 - output) * pq.T
            w1 = w1 - e * output * (1 - output) * w2 * pq.T * (1 - pq.T) * input
    # 输出最终结果
    print(w1)
    print(w2)
    print(output)


w1 = np.array([[0.1, 0.8], [0.4, 0.6]])  # 第一层权重参数
w2 = np.array([0.3, 0.9])  # 第二层权重参数
bp(w1, w2)

w1 = np.array([[0.8, 0.1], [0.6, 0.4]])  # 第一层权重参数
w2 = np.array([0.9, 0.3])  # 第二层权重参数
bp(w1, w2)

w1 = np.array([[0.4, 0.7], [0.1, 0.2]])  # 第一层权重参数
w2 = np.array([0.1, 0.1])  # 第二层权重参数
bp(w1, w2)

w1 = np.array([[0.4, 0.6], [0.8, 0.1]])  # 第一层权重参数
w2 = np.array([0.2, 0.4])  # 第二层权重参数
bp(w1, w2)

w1 = np.array([[0.1, 0.6], [0.2, 0.2]])  # 第一层权重参数
w2 = np.array([0.9, 0.3])  # 第二层权重参数
bp(w1, w2)

w1 = np.array([[0.8, 0.7], [0.6, 0.2]])  # 第一层权重参数
w2 = np.array([0.9, 0.3])  # 第二层权重参数
bp(w1, w2)

w1 = np.array([[0.8, 0.1], [0.6, 0.4]])  # 第一层权重参数
w2 = np.array([0.3, 0.3])  # 第二层权重参数
bp(w1, w2)

