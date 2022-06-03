import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_grad_enabled(True)
        
        self.first = nn.Linear(784, 256)
        self.second = nn.Linear(256, 128)
        self.third = nn.Linear(128, 64)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.first(x)))
        x = self.dropout(F.relu(self.second(x)))
        x = self.dropout(F.relu(self.third(x)))
        x = self.output(x)
        
        return x
    
model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images = images.view(images.shape[0], -1)
                logps = model.forward(images)
                test_loss += criterion(logps, labels)
                
                # ps = torch.exp(logps)
                top_p, top_class = logps.topk(1, dim=1)
                equals = (top_class == labels.view(*top_class.shape))
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                print(test_loss)
            model.train()
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
                
        print(f'Epoch: {e+1}/{epochs}')
        print(f'Training loss: {train_losses[-1]}')
        print(f'Testing loss: {test_losses[-1]}')
        print(f'Accuracy: {accuracy/len(testloader)}')

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.legend(frameon=False)
plt.show()
      
images, labels = next(iter(testloader))
model.eval()
for i in range(10):
    img = images[i].view(1, 784)

    with torch.no_grad():
        logits = model.forward(img)
        
    out = F.softmax(logits, dim=1)

    plt.imshow(img.reshape(28, 28))
    plt.show()

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    plt.barh([labels_map[i] for i in range(10)], out[0])
    plt.show()