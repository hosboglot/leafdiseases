import os

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

class LeavesDataset(Dataset):    
    def __init__(self, datapath, train, transform):
        super().__init__()

        if train:
            split_ratio = 0.8
        else:
            split_ratio = 0.2

        self.diseases = sorted(os.listdir(datapath))
        self.images = {}
        for i, disease in enumerate(self.diseases):
            files = np.array(os.listdir(os.path.join(datapath, disease)))
            files.sort()
            files = files.reshape(-1, 6)
            idxs = np.random.random_integers(0, len(files) - 1, int(len(files) * split_ratio))
            files = files[idxs]
            self.images[i] = files

        self.lens = np.array([len(val) for val in self.images.values()], dtype=np.int32)
        self.datapath = datapath
        self.transform = transform

    def __len__(self):
        return np.sum(self.lens)

    def __getitem__(self, idx):
        found = False
        disease = 0
        while not found:
            if idx >= np.sum(self.lens[:disease+1]):
                disease += 1
            else:
                found = True
        
        idx = int(idx - np.sum(self.lens[:disease]))

        images_paths = self.images[disease][idx]
        image = cv2.imread(os.path.join(self.datapath, self.diseases[disease], images_paths[0]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = (image.astype(np.float32)) / 255.0
        for i in range(1, len(images_paths)):
            img = cv2.imread(os.path.join(self.datapath, self.diseases[disease], images_paths[i]), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = (img.astype(np.float32)) / 255.0
            image = np.dstack((image, img))

        tensor_image = self.transform(image)
        tensor_label = torch.tensor(disease)

        return (tensor_image, tensor_label)
