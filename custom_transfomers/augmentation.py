import os
import math
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
from .augmentations import Augment, Cutout
from easydict import EasyDict


def create_config(config_file_exp):
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v
    return cfg


class CreatePairTransforms:
    def __init__(self, transformer1, transformer2):
        self.transformer1 = transformer1
        self.transformer2 = transformer2

    def __call__(self, x):
        return self.transformer1(x), self.transformer2(x)


class CreateTripleTransforms:
    def __init__(self, transformer1, transformer2, transformer3):
        self.transformer1 = transformer1
        self.transformer2 = transformer2
        self.transformer3 = transformer3

    def __call__(self, x):
        return self.transformer1(x), self.transformer2(x), self.transformer3(x)

class CreateQuarTransforms:
    def __init__(self, transformer1, transformer2, transformer3,transformer4):
        self.transformer1 = transformer1
        self.transformer2 = transformer2
        self.transformer3 = transformer3
        self.transformer4 = transformer4
    def __call__(self, x):
        return self.transformer1(x), self.transformer2(x), self.transformer3(x), self.transformer4(x)

def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        tf = [
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
        ]
        if p['normalization'] == True:
            tf.append(transforms.Normalize(**p['augmentation_kwargs']['normalize']))
        return transforms.Compose(tf)
    
    elif p['augmentation_strategy'] == 'scan':
        # Augmentation strategy from scan paper
        tf = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            ]
        if p['normalization'] == True:
            tf.append(transforms.Normalize(**p['augmentation_kwargs']['normalize']))
        tf.append(Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random']))
        return transforms.Compose(tf)
    
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    tf = [transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor()]
    if p['normalization'] == True:
        tf.append(transforms.Normalize(**p['transformation_kwargs']['normalize']))

    return transforms.Compose(tf)

