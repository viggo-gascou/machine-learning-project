from mlproject.helpers import data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import optim
import numpy as np
from torchvision import transforms
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train, X_test, y_train, y_test = data_loader(raw=False,scaled=True)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
# X_train, X_test = torch.tensor(X_train).float(), torch.tensor(X_test).float() # transform to torch tensor
# y_train, y_test = torch.tensor(y_train).float(), torch.tensor(y_test).float() # transform to torch tensor

class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""
    
    def __init__(self, x,y, transform = None):
        """Method to initilaize variables.""" 
        self.fashion_MNIST = x
        self.labels = y
        self.transform = transform
        
        label = [i for i in self.labels]
        image = [i for i in self.fashion_MNIST]
        

        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

train_set = FashionDataset(X_train,y_train, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(X_test,y_test, transform=transforms.Compose([transforms.ToTensor()]))

train_set, val_set = torch.utils.data.random_split(train_set, [8000, 2000])
train_loader = DataLoader(train_set, batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0)
val_loader = DataLoader(val_set, batch_size=100, shuffle=True, num_workers=0)
test_loader = DataLoader(train_set, batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0)


loaders = {
    'train' : train_loader,
    'validation'  : val_loader,
    'test'  : test_loader,
}


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              
                out_channels=6,            
                kernel_size=5,              
                padding=2,                  
            ),                              
            nn.Sigmoid(),                      
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), 
            nn.Flatten() 
        )
        #self.flatten = ,
        self.linear1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
        )
                     

        #self.dropout = nn.Dropout(0.25)
        # fully connected layer, output 5 classes
        self.fully_connected = nn.Linear(84,5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        #x = self.dropout(x)
        output = self.fully_connected(x)
        #x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #x = x.view(x.size(0), -1)       
        #output = self.fully_connected(x)
        return output    # return x for visualization

model = CNN()


loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)


def train(num_epochs, ann, loaders):
    min_valid_loss = np.inf
        
    if torch.cuda.is_available():
        ann.cuda()

    for epoch in range(num_epochs):

        # Train the model
        epoch_train_loss = 0
        # This line tells our ANN that it's in the training mode
        # This will become relevant when we introduce layers that behave
        # differently in training and deployment/evaluation modes
        ann.train()
        for i, (images, labels) in enumerate(loaders['train']):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            # forward pass
            output = ann(images) 
            
            loss = loss_f(output, labels)
            epoch_train_loss += loss.item()

            # clear gradients for this training step   
            optimizer.zero_grad()           

            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                

        # Validate the model
        epoch_val_loss = 0
        ann.eval()
        for images_v, labels_v in loaders['validation']:
            if torch.cuda.is_available():
                images_v, labels_v = images_v.cuda(), labels_v.cuda()


            output = ann(images_v)
            loss_v = loss_f(output, labels_v)
            epoch_val_loss += loss_v.item()

        print(f'Epoch {epoch+1}')
        print(f'Training Loss: {(epoch_train_loss / len(loaders["train"])):.3f}')
        print(f'Validation Loss: {(epoch_val_loss / len(loaders["validation"])):.3f}')
        print('-------------------')

        if min_valid_loss > epoch_val_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.3f}--->{epoch_val_loss:.3f}) \t Saving The Model')
            min_valid_loss = epoch_val_loss
            # Saving State Dict
            #torch.save(ann.state_dict(), 'saved_models/LeNet.pth')

train(5, model, loaders)


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
    print("Accuracy: {:.3f}% ".format( float(correct) / len(data_loader)))                           



evaluate(model, loaders['train'])