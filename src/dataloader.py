import cv2
import glob
import torch
import torchvision.transforms as transforms
import numpy
from scaleRGB import *
from torch.utils.data import Dataset


# Set img directory and tensor type
img_dir = "../face_images/*.jpg"
torch.set_default_tensor_type('torch.FloatTensor')

# Reads in data from the face_images folder and returns a numpy array containing the RGB channels
def getData():
    files = glob.glob(img_dir)
    data =[]
    for f1 in files:
        img = cv2.cvtColor(cv2.cv2.imread(f1), cv2.COLOR_BGR2RGB).astype(numpy.float32) * 1.0/255
        data.append(img)
    return numpy.array(data, dtype='f')

# Coverts numpy array to a Pytorch Tensor
def getTorchTensor(images): 
    return torch.from_numpy(images).permute(0,3,1,2)
    
# Transforms the images to augemnt the dataset by 10x
def dataAugment(imageTensor):
    # Create a composition of transforms
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p =.5),
        transforms.RandomApply([transforms.RandomResizedCrop(size=128)], p =.5),
        transforms.RandomApply([scaleRGB()], p =.5),
    ])
    # Apply transfroms and return augmented tensor of images
    img = data_transforms(imageTensor)
    img = torch.cat((imageTensor, img))
    for i in range(8):
        img2 = data_transforms(imageTensor)
        img = torch.cat((img, img2))
    
    return img

# Convert from RGB to LAB space
def convertSpace(imageTensor):
    data = []
    newImage = torch.empty(imageTensor.size())
    for i  in range(len(imageTensor)):
        data.append(cv2.cvtColor(imageTensor[i].permute(1,2,0).numpy(), cv2.COLOR_RGB2LAB))
    return getTorchTensor(numpy.array(data, dtype='f'))

# Get Tensor to split into labels and inputs
def getTrainTest(imageTensor):
    xtensor = torch.mul(imageTensor[:,:1,:,:], .01)
    ytensor = imageTensor[:,1:3,:,:]
    return xtensor, ytensor

# Derived class for Getting Image dataset
class CustomImageDataset(Dataset):
    def __init__(self, dataTensor):
        x,y = getTrainTest(dataTensor)
        self.img = x
        self.img_labels = y

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.img[idx], self.img_labels[idx]

# Get Tensor to split into labels and inputs
def getTrainTestSimple(imageTensor):
    xtensor = torch.mul(imageTensor[:,:1,:,:], .01)
    ytensor = imageTensor[:,1:3,:,:]
    for i in range(7500):
        ytensor[i, 0] = torch.mean(ytensor[i, 0])
        ytensor[i, 1] = torch.mean(ytensor[i, 1])
    ytensor.resize_((7500,2))    
    return xtensor, ytensor

# Derived class for Getting Image dataset
class CustomImageDatasetSimple(Dataset):
    def __init__(self):
        images = getData()
        ten = getTorchTensor(images)
        img = dataAugment(ten)
        img = convertSpace(img)
        x,y = getTrainTestSimple(img)
        self.img = x
        self.img_labels = y

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        return self.img[idx], self.img_labels[idx]

""" WRITES INFO TO FILE
INPUTS:
    name: name of file
    lines: info to write
RETURNS:
    NA
"""
def file_write(name, lines):
    with open(name,'w') as f:
        f.writelines(lines)