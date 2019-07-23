import torch
from tqdm import tqdm
from sklearn.svm import SVC
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

train_data = torch.load('mnist_feature_train')
test_data = torch.load('mnist_feature_test')

grid = make_grid(train_data[0][0][0:10].reshape(80,20,20).unsqueeze(1).data, nrow=8, normalize=True).permute(1, 2, 0).numpy()
plt.imshow(grid)
plt.show()
print(train_data[0][1][0:10])

index = np.random.randint(0, 60000, size=(5000))

def get_xy(data):
    x = None
    y = None
    for seg in tqdm(data):
        if x is None:
            x = seg[0]
            y = seg[1]
        else:
            x = torch.cat([x,seg[0]])
            y = torch.cat([y,seg[1]])

    return x, y

train_x, train_y = get_xy(train_data)
test_x, test_y = get_xy(test_data)

model = SVC(probability=True)
model.fit(train_x.numpy()[index], train_y.numpy()[index])
result = model.predict(test_x.numpy())
acc = len(result[result==test_y.numpy()])
print(acc)



