import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import torchvision.utils as vutils
from patchify import patchify 

def load_transforms(name):
    """Load data transformations.
    
    Note:
        - Gaussian Blur is defined at the bottom of this file.
    
    """
    _name = name.lower()
    if _name == "cifar_sup":
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.765625, 0.765625),ratio=(1., 1.)),
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.765625, 0.765625),ratio=(1., 1.)),
            transforms.ToTensor(),normalize])

    elif _name == "cifar_patch":
        normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        aug_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
            transforms.ToTensor(), normalize])
        
    elif _name == "cifar_simclr_norm":
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
            transforms.ToTensor(),normalize])
    
    elif _name == "cifar_byol":
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                    (32, 32),
                    scale=(0.2, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        baseline_transform = transforms.Compose([
#             transforms.RandomResizedCrop(32,scale=(0.765625, 0.765625),ratio=(1., 1.)),
            transforms.ToTensor(),normalize])

    else:
        raise NameError("{} not found in transform loader".format(name))
    return aug_transform, baseline_transform


    
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size, n_chan=3):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(n_chan, n_chan, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=n_chan)
        self.blur_v = nn.Conv2d(n_chan, n_chan, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=n_chan)
        self.k = kernel_size
        self.r = radias
        self.c = n_chan

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(self.c, 1)

        self.blur_h.weight.data.copy_(x.view(self.c, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(self.c, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
        
  
class Brightness_Jitter(object):
    def __init__(self, jitter_range=0.4):
        self.jitter_range = jitter_range
    
    def __call__(self,img):
#         print(img.shape)
        jitter_ratio = 1 - self.jitter_range/2 + self.jitter_range*torch.rand((1,1,1))
        return (img*jitter_ratio)

class ContrastiveLearningViewGenerator(object):
    def __init__(self,aug_transform, add_gaussian, num_patch = 4):
        self.aug_transform = aug_transform
        
        self.num_patch = num_patch
        self.add_gaussian = add_gaussian
      
    def __call__(self, x):
    
        add_gaussian = self.add_gaussian
        

        
    
        normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.ToTensor(),
            normalize
        ])
        
        """
        baseline_transform = transforms.Compose([
            #transforms.RandomResizedCrop(14,scale=(0.4375, 0.4375), ratio=(1,1)),
            transforms.RandomCrop(14),
            
            transforms.ToTensor()])
            
        """ 
        baseline_transform = transforms.Compose([
            transforms.ToTensor()])
            
        gaussian_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.08),
            normalize
        ])
            
        x_tensor = baseline_transform(x)
        #print(x_tensor.shape)    
        
        patched_images = (x_tensor.unfold(1, 14, 5).unfold(2, 14, 5)).permute(1,2,0,3,4).reshape(16, 3, 14, 14) 
        #patched_images = (x_tensor.unfold(1, 14, 18).unfold(2, 14, 18)).permute(1,2,0,3,4).reshape(4, 3, 14, 14) 
        
        #patched_images = patched_images[1:3]
        
        #print(patched_images.shape)
        #
        
     
        #patched_images = torch.from_numpy(patchify(x_tensor.numpy(), (3, 8, 8), step=8))
        #print(patched_images.shape) 
        
        if self.num_patch>1:
            #augmented_x = [aug_transform(baseline_transform(x)) for i in range(self.num_patch)]
            
            augmented_x = [aug_transform(patch) for patch in patched_images]
            
            
        else:
        
            baseline_transform = transforms.Compose([
                transforms.RandomCrop(14),
                transforms.ToTensor()]
            )
            augmented_x = aug_transform(baseline_transform(x))
            
            
        if add_gaussian:
            augmented_x.extend([gaussian_transform(x_patch) for x_patch in patched_images])
        else:
            
            pass
        
        return augmented_x
        #return [x_patch for x_patch in patched_images]
        
        
