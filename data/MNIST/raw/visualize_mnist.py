import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define the neural network architecture (this should match your original model)
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST test dataset
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()  # Convert to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose for displaying
    plt.show()

# Get some random test images
dataiter = iter(testloader)
images, labels = dataiter.next()

# Make predictions using the loaded model
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Print the ground truth and predictions
print('GroundTruth: ', ' '.join(f'{labels[j].item()}' for j in range(8)))
print('Predicted: ', ' '.join(f'{predicted[j].item()}' for j in range(8)))

# Show the images with their predicted labels
imshow(torchvision.utils.make_grid(images[:8]))
