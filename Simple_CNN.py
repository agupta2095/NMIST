import torch
import torch.nn as nn  #All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.datasets as ds  # Standard datasets
import torchvision.transforms as tforms # Transformations we can perform on our dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader #Gives easier dataset management and creates mini batches

#Create a simple CNN
class CNN(nn.Module):
    def _init__(self, in_channels, num_classes): #For MNIST, in_channels = 1, and num_classes = 10
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=5, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.Fc1 = nn.Linear(10*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper-parameters
input_size = 784
num_classes = 10
lr = 1e-3
batch_size = 32
num_epochs = 50

#Load Data
train_dataset = ds.MNIST(root='dataset/Simple_CNN', train=True, transform = tforms.ToTensor(), download =True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ds.MNIST(root='dataset/Simple_CNN', train=False, transform= tforms.ToTensor(), download =True)
test_dataset = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model = CNN()
