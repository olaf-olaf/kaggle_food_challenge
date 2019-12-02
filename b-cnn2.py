import torch
import torchvision
import torch.nn as nn
import torch.nn.functional
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image


# import bilinear_resnet
# import CUB_200


class BCNN(nn.Module):
    def __init__(self, num_classes=80, pretrained=False):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, self.num_classes)

        # Freeze all previous layers.
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
            def init_weights(layer):
                if type(layer) == torch.nn.Conv2d or type(layer) == torch.nn.Linear:
                    torch.nn.init.kaiming_normal_(layer.weight.data)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias.data, val=0)
            self.fc.apply(init_weights)
            self.trainable_params = [
                {'params': self.fc.parameters()}
            ]
        else:
            self.trainable_params = self.parameters()

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*self.num_classes.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self.num_classes)
        return X

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



base_lr = 0.001
batch_size = 64
num_epochs = 1
weight_decay = 1e-5
num_classes = 80

save_model_path = 'b-cnn'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    model = BCNN(num_classes, pretrained=True).to(device)

    # model_d = model.state_dict()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)


    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    train_set, test_set = create_train_validation('food-recognition-challenge/train_labels.csv')

    # Put the train data in a loader
    train_data = DatasetFood(train_set, transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True, num_workers=1)

    # # Put the validation data in a DataLoader
    test_data = DatasetFood(test_set, transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=False, num_workers=1)


    print('Start training the fc layer...')
    best_acc = 0.
    best_epoch = 0
    end_patient = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0.
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

            print('Epoch %d: Iter %d, Loss %g' % (epoch + 1, i + 1, loss))
        train_acc = 100 * correct / total
        print('Testing on test dataset...')
        test_acc = test_accuracy(model, test_loader)
        print('Epoch [{}/{}] Loss: {:.4f} Train_Acc: {:.4f}  Test1_Acc: {:.4f}'
              .format(epoch + 1, num_epochs, epoch_loss, train_acc, test_acc))
        scheduler.step(test_acc)
        if test_acc > best_acc:
            model_file = os.path.join(save_model_path, 'resnet34_CUB_200_train_fc_epoch_%d_acc_%g.pth' %
                                      (best_epoch, best_acc))
            end_patient = 0
            best_acc = test_acc
            best_epoch = epoch + 1

        else:
            end_patient += 1

        # If the accuracy of the 10 iteration is not improved, the training ends
        if end_patient >= 10:
            break
    print('After the training, the end of the epoch %d, the accuracy %g is the highest' % (best_epoch, best_acc))
    # Get accuracy on validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validationimages: %d %%' % (
        100 * correct / total))





def test_accuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)

            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
        model.train()
        return 100 * correct / total


def main():
    train()


if __name__ == '__main__':
    main()
