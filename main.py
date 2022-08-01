"""
 Project: Facial Emotion Recognition using CNN in PyTorch
 Author: Deyuan Qu, Sudip Dhakal, Dominic Carrillo
"""
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt


class facialDataset(data.Dataset):
    # Initialization
    def __init__(self, root):
        super(facialDataset, self).__init__()
        self.root = root
        dt_path = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[0])
        dt_label = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[1])
        self.path = np.array(dt_path)[:, 0]
        self.label = np.array(dt_label)[:, 0]

    # Reading facial image
    def __getitem__(self, item):
        # Reading img
        face_img = cv2.imread(self.root + '\\' + self.path[item])
        # Reading single-channel grayscale image
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Histogram equalization
        face_hist = cv2.equalizeHist(face_gray)
        # Pixel value normalization, our img is 48x48,
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.cuda.FloatTensor')
        label = self.label[item]
        return face_tensor, label

    # Get the number of samples in the dataset
    def __len__(self):
        return self.path.shape[0]


class facialCNN(nn.Module):
    # Initialization of the CNN network structure
    def __init__(self):
        super(facialCNN, self).__init__()

        # The first convolution and pooling
        self.conv1 = nn.Sequential(
            # Convolutional layer
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            # Normalized
            nn.BatchNorm2d(num_features=64),
            # Activation function
            nn.RReLU(inplace=True),
            # Maximum pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # The second convolution and pooling
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # The third convolution and pooling
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Parameter initialization
        self.conv1.apply(Gaussian_weights_init)
        self.conv2.apply(Gaussian_weights_init)
        self.conv3.apply(Gaussian_weights_init)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # Forward propagation function
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Flattened Data
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


# Parameters initialization
def Gaussian_weights_init(m):
    classname = m.__class__.__name__
    #  Searching string, if not found return -1, if not return -1 means that the string contains the character
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# Training model
def model_train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # Initial cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data and split the batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # Using facial CNN model
    cnn_model = facialCNN()
    # Adding cuda to device
    cnn_model.to(device)
    # Loss function
    loss_function = nn.CrossEntropyLoss()
    # Using SGD optimizer
    optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # Learning rate decay
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(epochs):
        # Recording loss rate
        loss_rate = 0
        # Learning rate decay
        # scheduler.step()

        # Training CNN model
        cnn_model.train()
        for images, labels in train_loader:
            # Using cuda
            images = images.cuda()
            labels = labels.cuda()
            # Set the gradients to zero
            optimizer.zero_grad()
            # Forward propagation
            output = cnn_model.forward(images)
            # Calculate loss function
            loss_rate = loss_function(output, labels)
            # Back propagation based on loss rate
            loss_rate.backward()
            # Update parameters
            optimizer.step()

        # Print out loss rate for each epoch
        print('After {} epochs , the loss_rate is : '.format(epoch + 1), loss_rate.item())
        # Evaluate cnn model
        cnn_model.eval()
        # Get the accuracy of training dataset and validation dataset, and print it out
        acc_train = model_validate(cnn_model, train_dataset, batch_size)
        acc_val = model_validate(cnn_model, val_dataset, batch_size)
        print('After {} epochs , the acc_train is : '.format(epoch + 1), acc_train)
        print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)

        # Save accuracy data into .txt file
        file1 = open('accuracy_train.txt', 'a')
        file1.write(str(acc_train))
        file1.write('\r')

        file2 = open('accuracy_val.txt', 'a')
        file2.write(str(acc_val))
        file2.write('\r')

        file3 = open('loss_rate of training dataset.txt', 'a')
        file3.write(str(loss_rate.item()))
        file3.write('\r')

    return cnn_model


# Verify the accuracy of the model on the dataset
def model_validate(cnn_model, dataset, batch_size):
    data_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in data_loader:
        pred = cnn_model.forward(images)
        pred_tmp = pred.cuda().data.cpu().numpy()
        pred = np.argmax(pred_tmp, axis=1)

        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc


def main():
    train_dataset = facialDataset(root='C:/Users/dqu/Desktop/CV_Project/train')
    val_dataset = facialDataset(root='C:/Users/dqu/Desktop/CV_Project/validation')
    # We can try to tune these parameters to improve the recognition accuracy
    # batch_size=128, epochs=200, learning_rate=0.05, wt_decay=1e-5
    cnn_model = model_train(train_dataset, val_dataset, batch_size=128, epochs=200, learning_rate=0.05, wt_decay=1e-5)

    # save model
    # torch.save(model, 'model_net.pkl')


if __name__ == '__main__':
    main()

