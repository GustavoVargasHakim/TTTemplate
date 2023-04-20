import math
import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from dataset.cifar_new import CIFAR_New
from dataset.visdatest import *

'''----------------------------------------Augmentation Zoo------------------------------------------'''
NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
cifar_train = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(*NORM)])
cifar_test = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(*NORM)])

visda_train = transforms.Compose([transforms.Resize((256, 256)),
                                  transforms.RandomCrop((224, 224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

visda_test = transforms.Compose([transforms.Resize((256, 256)),
                                 transforms.CenterCrop((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

augment_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

office_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

simclr_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


'''----------------------------------------Name of corruptions------------------------------------------'''
common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def prepare_test_data(args):
    if args.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            test_set = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=cifar_test)
        elif args.corruption in common_corruptions:
            test_set_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' % (args.corruption))
            test_set_raw = test_set_raw[(args.level - 1) * tesize: args.level * tesize]
            test_set = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=cifar_test)
            test_set.data = test_set_raw

        elif args.corruption == 'cifar_new':
            test_set = CIFAR_New(root=args.dataroot + '/CIFAR10.1', transform=cifar_test)
            permute = False
        else:
            raise Exception('Corruption not found!')

    elif args.dataset == 'cifar100':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            test_set = torchvision.datasets.CIFAR100(root=args.dataroot,
                train=False, download=False, transform=cifar_test)
        elif args.corruption in common_corruptions:
            test_set_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' % (args.corruption))
            test_set_raw = test_set_raw[(args.level - 1) * tesize: args.level * tesize]
            test_set = torchvision.datasets.CIFAR100(root=args.dataroot,
                train=False, download=True, transform=cifar_test)

            test_set.data = test_set_raw
    elif args.dataset == 'visda':
        test_set = VisdaTest(args.dataroot, transforms=visda_test)
    elif args.dataset == 'office':
        test_set = ImageFolder(root=args.dataroot + 'OfficeHomeDataset_10072016/' + args.category, transform=office_test)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    else:
        test_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    if args.distributed:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
            shuffle=(test_sampler is None), num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)
    else:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    return test_loader, test_sampler


def prepare_val_data(args):
    if args.dataset == 'visda':
        val_set = ImageFolder(root=args.dataroot + 'validation/', transform=visda_test)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    else:
        val_sampler = None
    if not hasattr(args, 'workers'):
        args.workers = 1
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
        shuffle=(val_sampler is None), num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    return val_loader, val_sampler


def prepare_train_data(args, contrastive=False):
    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=args.dataroot,
            train=True, download=False, transform=cifar_train)
        val_set = torchvision.datasets.CIFAR10(root=args.dataroot,
            train=False, download=False, transform=cifar_test)
    elif args.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=args.dataroot + 'train/',
            train=True, download=False, transform=cifar_train)
        val_set = torchvision.datasets.CIFAR100(root=args.dataroot,
            train=False, download=False, transform=cifar_test)
    elif args.dataset == 'visda':
        if contrastive:
            dataset = ImageFolder(root=args.dataroot + 'train/', transform=TwoCropTransform(simclr_transforms))
        else:
            dataset = ImageFolder(root=args.dataroot + 'train/', transform=visda_train)
        train_set, val_set = random_split(dataset, [106678, 45719], generator=torch.Generator().manual_seed(args.seed))
    elif args.dataset == 'office':
        dataset = ImageFolder(root=args.dataroot + 'OfficeHomeDataset_10072016/' + args.category, transform=augment_transforms)
        long = len(dataset)
        train_set, val_set = random_split(dataset, [math.floor(long * 0.8),
                                             math.floor(long * 0.2)], generator=torch.Generator().manual_seed(args.seed))
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if args.dataset == 'visda' or args.dataset == 'office':
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    else:
        train_sampler = None
        val_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.dataset == 'visda':
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
            shuffle=(val_sampler is None), num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    else:
        val_loader = None
    return train_loader, train_sampler, val_loader, val_sampler
