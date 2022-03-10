import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, base_dir=None, split=None, domain_list=None, transforms=None):
        super(Dataset, self).__init__()
        self.base_dir = base_dir
        self.split = split
        self.domain_list = domain_list
        self.transforms = transforms
        self.image_dir_list = []
        if isinstance(self.domain_list, list):
            for domain in domain_list:
                self.image_dir_list += [os.path.join(self.base_dir, self.split, domain, file_name) for file_name
                                        in os.listdir(os.path.join(self.base_dir, self.split, domain))]
        elif isinstance(self.domain_list, str):
            self.image_dir_list += [os.path.join(self.base_dir, self.split, self.domain_list, file_name) for file_name
                                        in os.listdir(os.path.join(self.base_dir, self.split, self.domain_list))]
        else:
            raise ValueError("The type of \'domain_list\' need to be \'list\' or \'str\', but got \'{}\'".format(type(self.domain_list).__name__))
    

    def __len__(self):
        return len(self.image_dir_list)
    

    def __getitem__(self, index):
        image_dir = self.image_dir_list[index]
        _, image_name = os.path.split(image_dir)
        image = np.load(image_dir)['image'].astype(np.float32)
        label = np.load(image_dir)['label'].astype(np.int64)

        if self.split == 'test' or self.split == 'val':
            sample = {'image': image, 'label': label}
            if self.transforms:
                sample = self.transforms(sample)

            return sample, image_name.replace('.npz', '')
        else:
            sample = {'image': image, 'label': label}
            if self.transforms:
                sample = self.transforms(sample)
            return sample

class CenterCrop(object):
    """
    Center Crop 2D Slices
    """
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='edge')
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='edge')
        
        (w, h) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {'image': image, 'label': label}
        

class RandomCrop(object):
    """
    Crop 2D Slices
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='edge')
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='edge')

        (w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label, 'domain_label': sample['domain_label']}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    """ Create Onehot label """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label' : onehot_label}


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1])
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
