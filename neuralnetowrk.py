import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5))])

#Download and load the training data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Download and load the test data
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

#Define feedforward neural network: inout layer, one hidden layer, output layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) #28x28 input size, 128 units in the hidden layer
        self.fc2 = nn.Linear(128, 64) #128 units input, 64 units in the second hidde layer
        self.fc3 = nn.Linear(64, 10) #64 units input, 10 output units (one for each digit)

    def forward(self, x):
        x = x.view(-1, 28 * 28) #flatten the inout tensor to a 1D tensor
        x = F.relu(self.fc1(x)) #Apply reLU activation to the first layer
        x = F.relu(self.fc2(x)) #Apply ReLU activation to the second layer
        x = self.fc3(x) #output layer (no activation here, I will use CrossEntropyLoss)
        return x
    
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels, in trainloader:
        optimizer.zero_grad() #Zero the parameter gradients

        outputs = model(images) #forward pass
        loss = criterion(outputs, labels) #compute the loss
        loss.backward() #backward pass (compute gradients)
        optimizer.step() #update the parameters

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

torch.save(model.state_dict(), 'mnist_model.pth')