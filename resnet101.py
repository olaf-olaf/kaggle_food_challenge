import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

NUM_CLASSES = 80

class DatasetFood(Dataset):
    def __init__(self, train_set, transform=None):

        self.data = train_set
        self.data['img_name'] = "food-recognition-challenge/train_set/train_set/"+ self.data['img_name']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data['img_name'][index]
        image = Image.open(img_name)
        label = self.data['label'][index] - 1
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_train_validation(file_path):
    data = pd.read_csv(file_path)
    train_set, validation_set = train_test_split(data, test_size=0.1)
    train_set.reset_index(inplace = True)
    validation_set.reset_index(inplace = True)
    return train_set, validation_set


# # Prepare data
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
train_set, validation_set = create_train_validation('food-recognition-challenge/train_labels.csv')

# Put the train data in a loader
train_dataset = DatasetFood(train_set, transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True, num_workers=1)

# # Put the validation data in a DataLoader
validation_dataset = DatasetFood(validation_set, transform)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64,shuffle=False, num_workers=1)

# Load resnet101 freeze all layers, and add one extra output layer
model = models.resnet101(pretrained=True)

# unfreeze linear layers
for module in model.modules():
    if module._get_name() != 'Linear':
        print('layer: ',module._get_name())
        for param in module.parameters():
            param.requires_grad_(False)
    elif module._get_name() == 'Linear':
        print('layer: ',module._get_name())
        for param in module.parameters():
            param.requires_grad_(True)
# for param in model.parameters():
#     param.requires_grad = False

# Added two linear layers with output of 80 classes and softmax activation.
model.fc = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, NUM_CLASSES),
                        nn.LogSoftmax(dim=1))



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# Put model on gpu if available
if torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Train model
epochs = 1
for epoch in range(epochs):# loop over the dataset multiple times
    print("IN EPOCH")
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print("BATCH", i)
        # get the inputs; data is a list of [inputs, labels]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs, labels = data
        inputs,labels = inputs.to(device),labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics AND SAVE MODEL
        running_loss += loss.item()
        epoch_loss += outputs.shape[0] * loss.item()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 99))
            running_loss = 0.0
            # print epoch loss
    print(epoch+1, epoch_loss / len(train_set))

torch.save(model.state_dict(), "first_resnet_model2.0.pl")

# Get accuracy on validation set
correct = 0
total = 0
with torch.no_grad():
    for data in validation_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the validationimages: %d %%' % (
    100 * correct / total))
