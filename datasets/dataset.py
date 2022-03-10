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
        # domain_label = np.array(self.domains[index])
        _, image_name = os.path.split(image_dir)
        image = np.load(image_dir)['image'].astype(np.float32)
        label = np.load(image_dir)['label'].astype(np.int64)

        if self.split == 'test' or self.split == 'val':
            # sample = {'image': image, 'label': label, 'domain_label': domain_label}
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

        # return {'image': image, 'label': label, 'domain_label': sample['domain_label']}
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
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        # return {'image': image, 'label': label, 'domain_label': sample['domain_label']}
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
        # return {'image': image, 'label': label, 'domain_label': sample['domain_label']}
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
        # return {'image': image, 'label': label, 'onehot_label' : onehot_label, 'domain_label': sample['domain_label']}
        return {'image': image, 'label': label, 'onehot_label' : onehot_label}


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1])
        if 'onehot_label' in sample:
            # return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
            #         'onehot_label': torch.from_numpy(sample['onehot_label']).long(),
            #         'domain_label': torch.from_numpy(sample['domain_label']).long()}
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            # return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
            #         'domain_label': torch.from_numpy(sample['domain_label']).long()}
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img )
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target


def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = np.fft.fftshift( amp_local )
    a_trg = np.fft.fftshift( amp_target )

    h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[h1:h2,w1:w2] = a_local[h1:h2,w1:w2] * ratio + a_trg[h1:h2,w1:w2] * (1- ratio)
    a_local = np.fft.ifftshift( a_local )
    return a_local

def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_ )
    local_in_trg = np.real(local_in_trg)

    return local_in_trg

# from torchvision import transforms
# from torch.utils.data import DataLoader
# from skimage.io import imsave
# import itertools
# num_classes = 2
# transforms = transforms.Compose([
#     # CenterCrop((256, 256)),
#     # RandomRotFlip(),
#     # RandomNoise(),
#     CreateOnehotLabel(2),
#     ToTensor()
# ])
# # train_domain_list = ['S_A', 'S_B', 'S_C', 'S_D', 'S_E']
# trainset1 = Dataset(base_dir='/data/ziqi/datasets/brats/npz_data', split='val', domain_list=['t2', 'aug1', 'aug2', 'aug3'], transforms=transforms)
# # trainset2 = Dataset(base_dir='/data/ziqi/datasets/brats/npz_data', split='val', domain_list='aug1', transforms=transforms)
# trainloader1 = DataLoader(trainset1, batch_size=8, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
# # trainloader2 = DataLoader(trainset2, batch_size=64, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
# if __name__ == '__main__':
#     # for i_batch, (domain_1, domain_2) in enumerate(zip(itertools.cycle(trainloader1), trainloader2)):
#     #     print(i_batch)
#     #     volume_batch, label_batch = domain_1[0]['image'], domain_1[0]['label']
#     #     onehot_batch = domain_1[0]['onehot_label']
#     #     print(onehot_batch.shape)
        
#     #     volume_batch, label_batch = domain_2[0]['image'], domain_2[0]['label']
#     #     onehot_batch = domain_2[0]['onehot_label']
#     #     print(onehot_batch.shape)
#     for i_idx, sample_batch in enumerate(trainloader1):
#         print(i_idx)
#         volume_batch, label_batch = sample_batch[0]['image'], sample_batch[0]['label']
#         onehot_batch = sample_batch[0]['onehot_label']
#         domain_batch = sample_batch[0]['domain_label']
#         # print(onehot_batch.shape)
#         print(domain_batch)
