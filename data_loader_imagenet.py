##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
# ADAPTED VERSION                                #
##################################################
import os, sys, hashlib, torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn import model_selection
from autoaugment import ImageNetPolicy

from cutout import Cutout

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    else:
        return check_md5(fpath, md5)


class ImageNet16(data.Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ["train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"],
        ["train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"],
        ["train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"],
        ["train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"],
        ["train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"],
        ["train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"],
        ["train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"],
        ["train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"],
        ["train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"],
        ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
    ]
    valid_list = [
        ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
    ]

    def __init__(self, root, train, transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            # print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert (
                isinstance(use_num_of_class_only, int)
                and use_num_of_class_only > 0
                and use_num_of_class_only < 1000
            ), "invalid use_num_of_class_only : {:}".format(use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets

    def __repr__(self):
        return "{name}({num} images, {classes} classes)".format(
            name=self.__class__.__name__,
            num=len(self.data),
            classes=len(set(self.targets)),
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.valid_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True



def get_train_valid_loader(data_dir,
                            batch_size,
                            augment=True,
                            shuffle=False,
                            show_sample=False,
                            num_workers=4,
                            pin_memory=False,
                            cutout=False,
                            cutout_length=16,
                            auto_augment=False,
                            resize=False,
                            datasetType='Full',
                            n_classes=120,
                            resizelength=300):

    datasetType = datasetType.lower()

    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )

    valid_transform = []
    if resize: # utility for creating the search space if models are from original imagenet
        valid_transform = [
            transforms.Resize(resizelength),
        ]
    valid_transform.extend([
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose(valid_transform)

    if resize:
        train_transform = [
            transforms.Resize(resizelength),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(resizelength, padding=4),
        ]
    else:
        train_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),   
        ]

    if auto_augment:
        train_transform.extend([
            ImageNetPolicy(),
        ])
    train_transform.extend([
        transforms.ToTensor(),
        normalize,
    ])
    if cutout:
        train_transform.extend([
            Cutout(cutout_length),
        ])

    train_transform = transforms.Compose(train_transform)

    '''
    if resize:
        train_transform = transforms.Compose([
            transforms.Resize(resizelength),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(resizelength, padding=4),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(16, padding=2),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize,
            ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length)) #can be changed
    '''
    #if auto_augment:
    #    train_transform.transforms.insert(2, AutoAugment())

    train_dataset = ImageNet16(data_dir, True , train_transform, 120) 
    val_dataset = ImageNet16(data_dir, False, valid_transform, 120) 


    if 'full' in datasetType or 'trainentire' in datasetType: #entire dataset
        train_loader  = torch.utils.data.DataLoader(train_dataset , batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers, pin_memory=pin_memory)
        val_loader  = torch.utils.data.DataLoader(val_dataset , batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, n_classes
    if 'partial' in datasetType and 'fly' in datasetType: #partialfly
        targets = train_dataset.targets
        train_idx, valid_idx = model_selection.train_test_split(
            np.arange(len(targets)), test_size=0.02, train_size=0.08, random_state=42, shuffle=True, stratify=targets)
    else: #partial
        with open(data_dir+'trainPartial10ImageNet16Indexes', 'rb') as f:
            train_idx = pickle.load(f)
        with open(data_dir+'valPartial10ImageNet16Indexes', 'rb') as f:
            valid_idx = pickle.load(f)

    train_loader  = torch.utils.data.DataLoader(train_dataset , batch_size=batch_size, shuffle=shuffle,
                                                sampler=SequentialSampler(train_idx),
                                                num_workers=num_workers, pin_memory=pin_memory)
    val_loader  = torch.utils.data.DataLoader(train_dataset , batch_size=batch_size, shuffle=shuffle,
                                                sampler=SequentialSampler(valid_idx),
                                                num_workers=num_workers, pin_memory=pin_memory)


    # To generate splits
    '''
    targets = train_dataset.targets
    train_idx, valid_idx = model_selection.train_test_split(
        np.arange(len(targets)), test_size=0.02, train_size=0.08, random_state=42, shuffle=True, stratify=targets)

    # Check stratification
    print(np.unique(np.array(targets)[train_idx], return_counts=True))
    print(np.unique(np.array(targets)[valid_idx], return_counts=True))


    with open(data_dir+'trainPartial10ImageNet16Indexes', 'wb') as f:
        pickle.dump(train_idx, f)
    with open(data_dir+'valPartial10ImageNet16Indexes', 'wb') as f:
        pickle.dump(valid_idx, f)
    '''
    
    return train_loader, val_loader, n_classes


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    resize=False,
                    resizelength=300):
    
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )

    # define transform
    if resize:
        transform = transforms.Compose([
            transforms.Resize(resizelength),
            transforms.ToTensor(),
            normalize,])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,])


    val_dataset = ImageNet16(data_dir, False, transform, 120) 
    data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return data_loader


# For testing purposes
if __name__ == '__main__':
    '''
    cutout = True
    cutout_length = 16
    resize=False
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )

    valid_transform = []
    if resize: # utility for creating the search space if models are from original imagenet
        valid_transform = [
            transforms.Resize(300),
        ]
    valid_transform.extend([
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose(valid_transform)

    if resize:
        train_transform = transforms.Compose([
            transforms.Resize(300),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(300, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(16, padding=2),
                transforms.ToTensor(),
                normalize,
            ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length)) #can be changed
    train = ImageNet16('./data/ImageNet16', True , train_transform, 120) 
    valid = ImageNet16('./data/ImageNet16', False, valid_transform, 120) 
    print ( len(train) )
    print ( len(valid) )
    image, label = train[111]
    print(label)
    
    train_loader  = torch.utils.data.DataLoader(train , batch_size=10, num_workers=2, pin_memory=True)

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0]
        img_size = inputs.shape # image size of the given problem
        break    
    print(f'Image size: {img_size}')
    exit()
    '''
    train_loader, val_loader, n_classes = get_train_valid_loader('./data/ImageNet16/', 10, datasetType='partial')
    for train_images, labels in train_loader:  
        sample_image = train_images[0]
        img_size = sample_image.shape # image size of the given problem
        print(labels[0].shape)
        break    
    print(f'Image size: {img_size}')
    '''
    trainX = ImageNet16('~/.torch/cifar.python/ImageNet16', True , None, 200)
    validX = ImageNet16('~/.torch/cifar.python/ImageNet16', False , None, 200)
    print ( len(trainX) )
    print ( len(validX) )
    '''
    
