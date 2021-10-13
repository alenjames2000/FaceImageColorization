import torch
import random 

class scaleRGB(object):

    def __call__(self, sample):
        scale = random.uniform(.6, 1)
        image = torch.mul(sample, scale)
        return image
        