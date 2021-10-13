from numpy.core.fromnumeric import size
import cv2
import os
import glob
import torch
import torchvision.transforms as transforms
import numpy
from scaleRGB import *
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset



img_dir = "../face_images/*.jpg"
torch.set_default_tensor_type('torch.FloatTensor')


def getData():
    files = glob.glob(img_dir)
    data =[]
    for f1 in files:
        img = cv2.cvtColor(cv2.cv2.imread(f1), cv2.COLOR_BGR2RGB).astype(numpy.float32) * 1.0/255
        data.append(img)
    return numpy.array(data, dtype='f')

def getTorchTensor(images): 
    return torch.from_numpy(images).permute(0,3,1,2)
    

def dataAugment(imageTensor):
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p =.6),
        transforms.RandomApply([transforms.RandomResizedCrop(size=128)], p =.5),
        transforms.RandomApply([scaleRGB()], p =.5),
    ])
    img = data_transforms(imageTensor)
    img = torch.cat((imageTensor, img))
    for i in range(8):
        img2 = data_transforms(imageTensor)
        img = torch.cat((img, img2))
    
    return img

def convertSpace(imageTensor):
    data = []
    newImage = torch.empty((7500,3,128,128))
    for i  in range(7500):
        data.append(cv2.cvtColor(imageTensor[i].permute(1,2,0).numpy(), cv2.COLOR_RGB2LAB))
    return getTorchTensor(numpy.array(data, dtype='f'))

def getTrainTest(imageTensor):
    xtensor = torch.mul(imageTensor[:,:1,:,:], .01)
    ytensor = imageTensor[:,1:3,:,:]
    for i in range(7500):
        ytensor[i, 0] = torch.mean(ytensor[i, 0])
        ytensor[i, 1] = torch.mean(ytensor[i, 1])
    ytensor.resize_((7500,2))    
    return xtensor, ytensor

class CustomImageDataset(Dataset):
    def __init__(self):
        images = getData()
        ten = getTorchTensor(images)
        img = dataAugment(ten)
        img = convertSpace(img)
        x,y = getTrainTest(img)
        self.img = x
        self.img_labels = y

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        return self.img[idx], self.img_labels[idx]

if __name__ == '__main__':
    images = getData()
    ten = getTorchTensor(images)
    img = dataAugment(ten)
    img = convertSpace(img)
    x,y = getTrainTest(img)

"""img1 = cv2.cvtColor(img[750].permute(1,2,0).numpy(), cv2.COLOR_LAB2BGR)
    cv2.imshow('h', img1)
    cv2.waitKey(0)"""