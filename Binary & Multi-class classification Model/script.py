# ================== Image Classification ================== #
from torchvision import datasets
import torchvision.transforms as transforms

train_dir = "/data/train"
train_dataset = ImageFolder(root = train_dir, transform = transforms.ToTensor())

classes = train_dataset.classes
print(classes)
# ['cat', 'dog']

print(train_dataset.class_to_idx)
# ['cat': 0, 'dog': 1]

# Building the Binary Image Classification (convolutional layer)
class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        # Create a convolutional layer with 3 channels, 16 output channels, kernel size of 3, stride of 1, and padding of 1.
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 112 * 112, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.fc1(self.flatten(x))
        x = self.sigmoid(x)
        return x

class MultiClassCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 112 * 112, num_classes)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.fc1(self.flatten(x))
        x = self.softmax(x)
        return x
    

# ================== Image Classification QUIZ ================== #
# PROMPT:
    # Create a convolutional layer with 3 channels, 16 output channels, kernel size of 3, stride of 1, and padding of 1.
    # Create a fully connected layer with an input size of 16x32x32 and a number of classes equal to 1; include only the values in the provided order (16*32*32, 1).
    # Create a sigmoid activation function.

    # Binary classification model
    # As a deep learning practitioner, one of your main tasks is training models for image classification. You often encounter binary classification, where you need to distinguish between two classes. To streamline your workflow and ensure reusability, you have decided to create a template for a binary image classification CNN model, which can be applied to future projects.
    # The package torch and torch.nn as nn have been imported. All image sizes are 64x64 pixels.

class BinaryImageClassifier(nn.Module):
    def __init__(self):
        super(BinaryImageClassifier, self).__init__()
        
        # Create a convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Create a fully connected layer
        self.fc = nn.Linear(16*32*32, 1)

        # Create an activation function
        self.sigmoid = nn.Sigmoid()


# ================== Image Classification QUIZ ================== #
# PROMPT:
    # Define the __init__ method including self and num_classes as parameters.
    # Create a fully connected layer with the input size of 16*32*32 and the number of classes num_classes as output.
    # Create an activation function softmax with dim=1.

    # With a template for a binary classification model in place, you can now build on it to design a multi-class classification model. The model should handle different numbers of classes via a parameter, allowing you to tailor the model to a specific multi-class classification task in the future.
    # The packages torch and torch.nn as nn have been imported. All image sizes are 64x64 pixels.

class MultiClassImageClassifier(nn.Module):
    # Define the init method
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Create a fully connected layer
        self.fc = nn.Linear(16*32*32, num_classes)
        
        # Create an activation function
        self.softmax = nn.Softmax(dim = 1)


# ================== CONVOLUTIONAL LAYERS FOR IMAGES | DEEP LEARNING FOR IMAGES WITH PYTORCH ================== #
from torchvision.transforms import functional
image = PIL.Image.open('images/brain_MR.jpg')
num_channels = functional.pil_to_tensor(image)
print("Number of Channels", num_channels)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)

        conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)