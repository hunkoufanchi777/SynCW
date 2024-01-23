import os
import glob
from shutil import move
from os import listdir, rmdir

import torch
import torchvision
import torchvision.transforms as transforms


# Based on https://github.com/alecwangcq/GraSP/blob/master/utils/data_utils.py
def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'mnist':
        # transforms.Normalize((0.1307,), (0.3081,))
        t = transforms.Normalize((0.5,), (0.5,))
        transform_train = transforms.Compose([transforms.ToTensor(), t])
        transform_test = transforms.Compose([transforms.ToTensor(), t])

    if dataset == 'fashionmnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if dataset == 'cinic-10':
        # cinic_directory = '/path/to/cinic/directory'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

    if dataset == 'tiny_imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


# Based on https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
def TINYIMAGENET(root, train=True, transform=None, target_transform=None, download=False):
    def _exists(root, filename):
        return os.path.exists(os.path.join(root, filename))

    def _download(url, root, filename):
        torchvision.datasets.utils.download_and_extract_archive(url=url,
                                                                download_root=root,
                                                                extract_root=root,
                                                                filename=filename)

    def _setup(root, base_folder):
        # 将val集数据像ImageNet一样放在以类别命名的文件夹，以符合pytorch读取数据的要求
        target_folder = os.path.join(root, base_folder, 'val/')

        val_dict = {}
        with open(target_folder + 'val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]

        paths = glob.glob(target_folder + 'images/*')

        paths[0].split('/')[-1]
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(target_folder + str(folder)):
                os.mkdir(target_folder + str(folder))
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            dest = target_folder + str(folder) + '/' + str(file)
            move(path, dest)
        os.remove(target_folder + 'val_annotations.txt')
        rmdir(target_folder + 'images')

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'

    if download and not _exists(root, filename):
        _download(url, root, filename)
        _setup(root, base_folder)
    folder = os.path.join(root, base_folder, 'train' if train else 'val')

    return torchvision.datasets.ImageFolder(folder, transform=transform, target_transform=target_transform)


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=2, root='../Data', trainset_shuffle=True):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=root + '/mnist', train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.MNIST(root=root + '/mnist', train=False, download=True, transform=transform_test)

    if dataset == 'fashionmnist':
        trainset = torchvision.datasets.FashionMNIST(root=root + '/FashionMNIST', train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root=root + '/FashionMNIST', train=False, download=True, transform=transform_test)

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root + '/cifar-10-python', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root + '/cifar-10-python', train=False, download=True,
                                               transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cinic-10':
        trainset = torchvision.datasets.ImageFolder(root + '/cinic-10/trainval', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + '/cinic-10/test', transform=transform_test)

    if dataset == 'tiny_imagenet':
        num_workers = 16
        # trainset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/train', transform=transform_train)
        # testset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/val', transform=transform_test)
        trainset = TINYIMAGENET(root, train=True, download=True, transform=transform_train)
        testset = TINYIMAGENET(root, train=False, download=True, transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=trainset_shuffle,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader
