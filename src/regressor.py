import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import *
from torch.utils.data import DataLoader

# Derived class for CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, 2, 2)
        self.conv2 = nn.Conv2d(3, 3, 5, 2, 2)
        self.conv3 = nn.Conv2d(3, 3, 5, 2, 2)
        self.conv4 = nn.Conv2d(3, 3, 5, 2, 2)
        self.conv5 = nn.Conv2d(3, 3, 5, 2, 2)
        self.conv6 = nn.Conv2d(3, 3, 5, 2, 2)
        self.conv7 = nn.Conv2d(3, 2, 5, 2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        return torch.reshape(x, (len(x), 2))

# Create NN
net = Net()


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Set up dataloaders
training_data = CustomImageDatasetSimple()
trainloader = DataLoader(training_data, batch_size=64, shuffle=True)

comp = []
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        comp = (f'Prediction:{outputs}\nCorrect:{labels}\n')
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

file_write(r"../Predictions/MeanChrominance/Predictions.txt", comp)