import torch
import random 

# Custom transform to scale RGB channels
class scaleRGB(object):

    def __call__(self, sample):
        scale = random.uniform(.6, 1)
        image = torch.mul(sample, scale)
        return image
        