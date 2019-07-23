from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model
import random
import numpy as np
import matplotlib.pyplot as plt

n = 10000
test_n = 100

def con2bin(x):
    x_list = []
    sx = '{:08b}'.format(x)
    for bi in sx:
        x_list.append(int(bi))
    return x_list

def bin2ten(x):
    weight = [2**7,2**6,2**5,2**4,2**3,2**2,2**1,2**0]
    index = np.where(x>0.5)
    new_x = np.zeros_like(x)
    new_x[index] = 1
    sum = new_x.dot(weight)
    return int(sum)

def get_data(n):
    data_x = []
    data_y = []
    for i in range(n):
        x1 = random.randint(0, 127)
        x2 = random.randint(0, 127)
        y = x1 + x2
        cx1 = con2bin(x1)
        cx2 = con2bin(x2)
        cy = con2bin(y)
        print(cy)
        data_x.append([cx1, cx2])
        data_y.append(cy)

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    return data_x, data_y

data_x, data_y = get_data(n)
test_x, test_y = get_data(test_n)

print(data_x.shape)
print(data_y.shape)
print(test_x.shape)
print(test_y.shape)

model = Sequential()
model.add(LSTM(256, activation='relu',input_shape=(2, 8)))
model.add(Dense(8, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
LSTM = model.fit(data_x, data_y, epochs=100, batch_size=128, validation_data=(test_x, test_y), verbose=2, shuffle=False)

plt.plot(LSTM.history['loss'])
plt.plot(LSTM.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.grid()
plt.show()

plt.plot(LSTM.history['acc'])
plt.plot(LSTM.history['val_acc'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.grid()
plt.show()

plot_model(model, to_file=r'./model.png', show_shapes=True)

for i in range(test_n):
    x1 = random.randint(0, 128)
    x2 = random.randint(0, 128)
    y = x1 + x2
    cx1 = con2bin(x1)
    cx2 = con2bin(x2)
    cy = con2bin(y)
    input_x = np.array([cx1,cx2])
    pre = model.predict(input_x.reshape(1,2,8))
    pre = bin2ten(pre.reshape(-1))
    print('num{} : {} + {} = {}, lstm({},{}) = {}'.format(i,x1,x2,y,x1,x2,pre))

model.save('lstm_model.h5')


