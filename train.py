from pathlib import Path
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader
from torchvision import transforms

from leaves_dataset import LeavesDataset

class ConvNet(nn.Module): 
    def __init__(self, num_classes): 
        super(ConvNet, self).__init__() 
        self.layer_conv1 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ) 
        self.layer_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ) 
        self.layer_dropout = nn.Dropout()
        self.layer_fc1 = nn.Linear(262144, 1000)
        self.layer_fc2 = nn.Linear(1000, num_classes)

    def forward(self, x): 
       out = self.layer_conv1(x) 
       out = self.layer_conv2(out)
       out = out.reshape(out.size(0), -1)
       out = self.layer_dropout(out)
       out = self.layer_fc1(out)
       out = self.layer_fc2(out)
       return out


def train_model(data_path, ckpt_path, num_classes = 4, batch_size = 16, num_epochs = 10, learning_rate = 0.00001, random_seed=7):
    np.random.seed(random_seed)
    device = 'cuda:0'

    trans = transforms.Compose([transforms.ToTensor()])
    train_dataset = LeavesDataset(data_path, train=True, transform=trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)

    model = ConvNet(num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for (images, labels) in tqdm(train_loader, desc='Epoch: ' + str(epoch+1)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / float(total) * 100)

            del images
            del labels
            # torch.cuda.empty_cache()
        
        print('Loss: {:.4f}, Accuracy: {:.2f}%'.format(loss.item(), (correct / float(total)) * 100))

    torch.save(model.state_dict(), ckpt_path)
    
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(loss_list, color=[1, 0, 0], label='Loss')
    axs[1].plot(acc_list, color=[0, 0, 1], label='Accuracy')
    fig.legend()
    fig.savefig(Path(ckpt_path).stem + '_plots.png')

def File(f):
    path = Path(f)
    if not Path(f).is_file():
        raise argparse.ArgumentTypeError()
    return path

if __name__ == '__main__':
    train_model('processed_images/Grape', 'checkpoints/Grape.ckpt')
    # train_model('processed_images/Apple', 'checkpoints/Apple.ckpt')
    # parser = argparse.ArgumentParser(description='Train')
    # parser.add_argument('source', type=Path, help='path to plants dataset')
    # parser.add_argument('ckpt', type=File, help='path to checkpoint')
    # args = parser.parse_args()

    # train_model(args.source, args.ckpt)
