import torch
import random 

# Check if CUDA is avaialable
global device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom transform to scale RGB channels
class scaleRGB(object):

    def __call__(self, sample):
        scale = random.uniform(.6, 1)
        image = torch.mul(sample, scale)
        return image
        