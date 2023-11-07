#### Dataset Preparation

import torch
import os
import numpy as np
import cv2
import torchvision.transforms as T

from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, src_dir, csv, train, test):
        self.src_dir = src_dir  ### Source Dir is added
        self.csv = csv
        self.train = train
        self.test = test
        self.images = self.csv[:2000]['Path']   ## self.all_image_names
        self.labels = np.array(self.csv.drop(['Path', 'Classes', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA' ], axis=1)) ### self.all_labels
        self.train_ratio = int(0.9 * len(self.csv[:2000]))
        self.val_ratio = len(self.csv[:2000]) - self.train_ratio
        
        #Set the training images and labels
        
        if self.train == True:
            print("Total No. of Training images are: {}".format(self.train_ratio))
            self.image_names = list(self.images[:self.train_ratio])
            self.label_names = list(self.labels[:self.train_ratio])   ## self.labels
            
            # Define training augmentations
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop(384),
                T.RandomRotation(degrees=(-15,15)),
                T.RandomAutocontrast(p=0.5),
                T.RandomEqualize(p=0.5),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229,0.224, 0.225)),
            ])       
            
        elif self.train == False and self.test == False:
            print("Total No. of Validation images are: {}".format(self.val_ratio))
            self.image_names = list(self.images[-self.val_ratio:-100])
            self.label_names = list(self.labels[-self.val_ratio:-100])
            
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(384),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229,0.224, 0.225)),
            ])
            
        elif self.test == True and self.train == False:
            print('Total No. of Test images are: 100')
            self.image_names = list(self.images[-100:])
            self.label_names = list(self.labels[-100:])
            
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(384),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229,0.224, 0.225)),
            ])
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        imagepath = os.path.join (self.src_dir, self.image_names[index])
        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        targets = self.label_names[index]
        
        return {
            'image': torch.tensor(image).to(torch.float32),
            'label': torch.tensor(targets).to(torch.float32)
            #'label': torch.tensor(targets, dtype=torch.float32)
        }
