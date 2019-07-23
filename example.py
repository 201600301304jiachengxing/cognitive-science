import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle

import hmax
from tqdm import tqdm

# load
batch_size = 16
print('Data load')
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model with the universal patch set
print('Constructing model')
model = hmax.HMAX('./universal_patch_set.mat')

# Determine whether there is a compatible GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Run the model on the example images
print('Running model on', device)
model = model.to(device)
c2x = None
c2y = None
for (X, y) in tqdm(train_loader):
    if c2x is None:
        c2x = model(X.to(device))
        c2y = y
    else:
        c2x = torch.cat([c2x, model(X.to(device))])
        c2y = torch.cat([c2y, y])


c2x_t = None
c2y_t = None
for (X, y) in tqdm(train_loader):
    if c2x_t is None:
        c2x_t = model(X.to(device))
        c2y_t = y
    else:
        c2x_t = torch.cat([c2x_t, model(X.to(device))])
        c2y_t = torch.cat([c2y_t, y])


print('Saving output c2 to: c2.pt')
result = {'x':c2x, 'y':c2y, 'xt':c2x_t, 'yt':c2y_t}
torch.save(result, 'c2.pt')
print('done')