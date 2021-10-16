import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import *
from torch.utils.data import DataLoader

# Derived class for CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(1, 3, 5, 2, 2)
        self.conv2 = nn.Conv2d(3, 3, 5, 2, 2)
        #self.conv3 = nn.Conv2d(3, 3, 5, 2, 2)
        #self.conv4 = nn.Conv2d(3, 3, 5, 2, 2)
        #self.conv5 = nn.Conv2d(3, 3, 5, 2, 2)
        #self.conv6 = nn.ConvTranspose2d(3, 3, 2, 2)
        #self.conv7 = nn.ConvTranspose2d(3, 3, 2, 2)
        #self.conv8 = nn.ConvTranspose2d(3, 3, 2, 2)
        self.conv9 = nn.ConvTranspose2d(3, 3, 2, 2)
        self.conv10 = nn.ConvTranspose2d(3, 2, 2, 2)
        

    def forward(self, x):
        x = F.relu(self.bm(self.conv1(x)))
        x = F.relu(self.bm(self.conv2(x)))
        #x = F.relu(self.bm(self.conv3(x)))
        #x = F.relu(self.bm(self.conv4(x)))
        #x = F.relu(self.bm(self.conv5(x)))
        #x = F.relu(self.conv6(x))
        #x = F.relu(self.conv7(x))
        #x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        return x

# Set up dataloaders
images = getData()
ten = getTorchTensor(images)
train, test = torch.split(ten, [int(.9*len(ten)), int(.1*len(ten))])
train = train
train = dataAugment(train)
train = convertSpace(train)
test = convertSpace(test)

training_data = CustomImageDataset(train)
testing_data = CustomImageDataset(test)
trainloader = DataLoader(training_data, batch_size=64, shuffle=True)
testloader = DataLoader(testing_data, batch_size=len(testing_data), shuffle=True)

# Create NN
net = Net()
net.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # COnvert them to CUDA tensors
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Test Model with MSE
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)
net.eval()
outputs = net(images)
print(f'MSE: {criterion(outputs, labels).item()}')