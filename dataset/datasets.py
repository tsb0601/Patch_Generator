import os
import numpy as np
import torchvision
from torch.utils.data import ConcatDataset
from .aug import load_transforms, ContrastiveLearningViewGenerator


def load_dataset(data_name, transform_name=None, use_baseline=False, train=True, into_patches=False, path="./data/"):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    
    Parameters:
        data_name (str): name of the dataset
        transform_name (torchvision.transform): name of transform to be applied (see aug.py)
        use_baseline (bool): use baseline transform or augmentation transform
        train (bool): load training set or not
        contrastive (bool): whether to convert transform to multiview augmentation for contrastive learning.
        n_views (bool): number of views for contrastive learning
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    aug_transform, baseline_transform = load_transforms(transform_name)
    transform = baseline_transform if use_baseline else aug_transform 
    if into_patches:
        transform = ContrastiveLearningViewGenerator(transform)
        
    _name = data_name.lower()
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "CIFAR10"), train=train, download=True, transform=transform)
        trainset.num_classes = 10

    else:
        raise NameError("{} not found in trainset loader".format(_name))
    return trainset

def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                       9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                      12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                       4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                       9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                      19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    return np.array(coarse_targets)[targets]