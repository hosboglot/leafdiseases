from pathlib import Path
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from leaves_dataset import LeavesDataset
from train import ConvNet


def test_model(data_path, ckpt_path, num_classes = 4, batch_size = 16):
    device = 'cuda:0'
    trans = transforms.Compose([transforms.ToTensor()])

    test_dataset = LeavesDataset(data_path, train=False, transform=trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet(num_classes)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for (images, labels) in tqdm(test_loader, desc='eval'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            del images
            del labels

        print('Test Accuracy of the model: {} %'.format((correct / float(total)) * 100))


def File(f):
    path = Path(f)
    if not Path(f).is_file():
        raise argparse.ArgumentTypeError()
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('source', type=Path, help='path to plants dataset')
    parser.add_argument('ckpt', type=File, help='path to checkpoint')
    args = parser.parse_args()

    test_model(args.source, args.ckpt)