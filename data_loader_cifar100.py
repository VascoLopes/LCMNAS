import torch
import numpy as np
from sklearn import model_selection

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from cutout import Cutout
from autoaugment import CIFAR10Policy

import pickle

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment=True,
                           shuffle=False,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=True,
                           cutout=False,
                           cutout_length=16,
                           auto_augment=False,
                           resize=False,
                           datasetType='Full',
                           resizelength=300):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]
    normalize = transforms.Normalize(
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
    )

    # define transforms
    if resize:
        valid_transform = transforms.Compose([
            transforms.Resize(resizelength),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
        ])

    if resize:
        train_transform = [
            transforms.Resize(resizelength),
            transforms.RandomCrop(resizelength, padding=4),
        ]
    else:
        train_transform = [
            transforms.RandomCrop(32, padding=4),
        ]
    train_transform.extend([
        transforms.RandomHorizontalFlip(),
    ])
    if auto_augment:
        train_transform.extend([
            CIFAR10Policy(),
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
            transforms.RandomCrop(resizelength, padding=4),
            transforms.RandomHorizontalFlip(), CIFAR10Policy(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), CIFAR10Policy(),
            transforms.ToTensor(),
            normalize,
        ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length)) #can be changed
    '''
        
    '''
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    '''
    
    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )


    # Generate stratified splits, and store indexes
    '''
    targets = train_dataset.targets
    
    print (len(targets))
    train_idx, valid_idx = model_selection.train_test_split(
        np.arange(len(targets)), test_size=0.02, train_size=0.08, random_state=42, shuffle=True, stratify=targets)
    
    # Check stratification
    print(np.unique(np.array(targets)[train_idx], return_counts=True))
    print(np.unique(np.array(targets)[valid_idx], return_counts=True))

    with open('./data/cifar-100/trainPartial10Cifar100Indexes', 'wb') as f:
        pickle.dump(train_idx, f)
    with open('./data/cifar-100/valPartial10Cifar100Indexes', 'wb') as f:
        pickle.dump(valid_idx, f)
    '''
    


    if datasetType.lower() == 'full':
        with open(data_dir+'trainFullCifar100Indexes', 'rb') as f:
            train_idx = pickle.load(f)
        with open(data_dir+'valFullCifar100Indexes', 'rb') as f:
            valid_idx = pickle.load(f)

    elif "trainentire" in datasetType.lower():
        return (get_entire_train(train_dataset, batch_size, shuffle, num_workers, pin_memory, data_dir,resize,resizelength))

    elif "partial" in datasetType.lower() and "fly" in datasetType.lower():
        targets = train_dataset.targets
        train_idx, valid_idx = model_selection.train_test_split(
        np.arange(len(targets)), test_size=0.02, train_size=0.08, random_state=42, shuffle=True, stratify=targets)
        #print(len(set(train_idx)-set(valid_idx)))

    else:#Partial
        with open(data_dir+'trainPartial10Cifar100Indexes', 'rb') as f:
            train_idx = pickle.load(f)
        with open(data_dir+'valPartial10Cifar100Indexes', 'rb') as f:
            valid_idx = pickle.load(f)


    # Datasets are already shuffled using scikit to create the indexes
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        sampler=SequentialSampler(train_idx),
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle, 
        sampler=SequentialSampler(valid_idx),
        num_workers=num_workers, pin_memory=pin_memory,
    )


    ''' 
    # Full Dataset, normal 
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    '''

    n_classes = 100
    return (train_loader, valid_loader, n_classes)

def get_entire_train(train_dataset, batch_size, shuffle, num_workers, pin_memory, data_dir, resize=False, resizelength=300):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    normalize = transforms.Normalize(
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
    )
    valid_transform = []
    # define transforms
    if resize:
        valid_transform = [
            transforms.Resize(resizelength),
        ]
    valid_transform.extend([
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose(valid_transform)
    
    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=valid_transform,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory,
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    #print(len(train_loader) * batch_size)
    #print(len(valid_loader)* batch_size)

    n_classes = 100
    return (train_loader, valid_loader, n_classes)

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    resize=False,
                    resizelength=300):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    normalize = transforms.Normalize(
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
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


    dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=transform,
    )


    '''
    # Generate partial test dataset
    targets = dataset.targets
    _, test_idx = model_selection.train_test_split(
        np.arange(len(targets)), test_size=0.1, random_state=42, shuffle=True, stratify=targets)

    with open('./data/testPartial10Cifar100Indexes', 'wb') as f:
        pickle.dump(test_idx, f)

    print(np.unique(np.array(targets)[test_idx], return_counts=True))
    '''


    # Full test dataset
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader


'''
def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)
'''


if __name__ == "__main__":
    # Tests
    trainloader, valloader, n_classes = get_train_valid_loader("./data/cifar-100/", 10, \
                datasetType="PartialFly")
    