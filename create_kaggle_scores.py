import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image



# NOTES
# Create train / val test_set
# Lower learning rate when it stops decreasing. learning rate scheduler.
# Remove relu from last layer
class TestSetFood(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.data['img_name'] = "food-recognition-challenge/test_set/test_set/"+ self.data['img_name']
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

train_dataset = TestSetFood('food-recognition-challenge/sample.csv', transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50,shuffle=False, num_workers=2)


# Define whatever model you load here.
model = models.resnet101(pretrained=True)
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 80),
                                 nn.LogSoftmax(dim=1))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('first_resnet_model.pl', map_location=device))
model.eval()

def feed_forward():
    prediction_labels = []
    for i, data in enumerate(train_loader, 0):
        print("BATCH", i)
        # get the inputs; data is a list of [inputs, labels]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)
        output = model(inputs)
        _, predicted = output.max(dim=1)
        for pred in predicted:
            print(pred)
            prediction_labels.append(pred.item() + 1)
    return prediction_labels

def create_submission_csv(file_path, prediction_labels, submission_name):
    submission_csv = pd.read_csv(file_path)
    submission_csv['label'] = prediction_labels
    submission_csv.to_csv('submission_name')

prediction_labels = feed_forward()
create_submission_csv('food-recognition-challenge/sample.csv', prediction_labels, 'first_resnet_submision.csv')
