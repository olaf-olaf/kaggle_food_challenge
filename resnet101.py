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

USE_GPU = True
NUM_CLASSES = 80

class DatasetFood(Dataset):
    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)
        self.data['img_name'] = "/kaggle/input/food-recognition-challenge/train_set/train_set/"+ self.data['img_name']
        print("LENGTH DATASET", len(self.data['img_name']))
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


transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = DatasetFood('/kaggle/input/food-recognition-challenge//train_labels.csv', transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True, num_workers=2)



# Load resnet101 freeze all layers, and add one extra output layer
model = models.resnet101(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, NUM_CLASSES),
                                 nn.LogSoftmax(dim=1))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)


# Try to use a gpu if available
if USE_GPU and torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')

# Define training
def train_model(epochs = 1):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            print("BATCH", i)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics AND SAVE MODEL
            running_loss += loss.item()
            if i % 50 == 54:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                torch.save(model.state_dict(), "resnet_saves/model_iteration"+str(i)+"epoch"+str(epoch)+".pl")
        print('Finished Training')
train_model()

# SAVE LAST model
torch.save(model.state_dict(), "first_resnet_model.py")
print("FINAL MODEL SAVED")
