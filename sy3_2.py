import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the saved model weights (ensure the model path is correct)
model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))  # Ensure the model path is correct
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Directory containing the MNIST dataset
mnist_dir = r'D:/python/JiQiXueXi/MNIST'  # Update with your dataset location

# Download and load MNIST dataset (set download=False to avoid downloading)
train_dataset = datasets.MNIST(root=mnist_dir, train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root=mnist_dir, train=False, transform=transform, download=False)

# DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# List to store images and predictions
images, preds = [], []

# Load, preprocess, and predict a few images
for image, label in train_loader:
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    images.append(image)  # Store images

    # Iterate over each element in the batch and append its prediction
    for p in predicted:
        preds.append(p.item())  # Store each prediction

    # Visualize the images and predictions
    fig, axes = plt.subplots(1, len(images[0]), figsize=(15, 1.5))  # Create axes for the batch size
    for i, image in enumerate(images[0]):
        img = image.numpy().squeeze()  # Get the image and remove batch dimension
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Pred: {preds[i]}')
        axes[i].axis('off')
    plt.show()
    break  # Exit after showing the first batch of images
