from mlproject.helpers import data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import optim, tensor
import numpy as np
from torchvision import transforms
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train, X_test, y_train, y_test = data_loader(raw=False,scaled=True)

# Training
X_train = tensor(X_train).float()
y_train = tensor(y_train).long()

# Test
X_test = tensor(X_test).float()

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# We first need to define class with the following
# structure that essentially serves as an iterator
# over the given dataset
class ModelDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

# Next, we define the dataloader, note that
# you can also define the batch size and whether
# to shuffle the data after each epoch
training = ModelDataset(X_train,y_train)
trainloader = DataLoader(training, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 4)
        self.fc2 = nn.Linear(4, 5)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

clf = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(clf.parameters(), lr=0.1)

epochs = 20
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = clf(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.8f}')
            running_loss = 0.0

print('Finished Training')

def evaluate(model, data_loader):
    if torch.cuda.is_available():
        model.cuda()
        model.eval()
    correct = 0 
    for test_imgs, test_labels in data_loader:
        if torch.cuda.is_available():
            test_imgs, test_labels = test_imgs.cuda(), test_labels.cuda()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Accuracy: {:.3f}% ".format( float(correct) / len(data_loader)*100))                           



evaluate(clf, trainloader)