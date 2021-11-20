import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017, Butterfly200
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)


def get_loader(args):
    if args['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()

    if args['dataset'] == 'CUB_200_2011':

        image_size = 448 if args['model_type'] == 'ViT-B_16' else 384
        resize_size = 600 if args['model_type'] == 'ViT-B_16' else 500

        train_transform = transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                                              transforms.RandomCrop((image_size, image_size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_transform = transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                                             transforms.CenterCrop((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset = CUB(root=args['data_root'], is_train=True, transform=train_transform)
        testset = CUB(root=args['data_root'], is_train=False, transform=test_transform)

    elif args['dataset'] == 'butterfly200':

        image_size = 448 if args['model_type'] == 'ViT-B_16' else 384
        resize_size = 600 if args['model_type'] == 'ViT-B_16' else 500
        train_transform = transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                                              transforms.RandomCrop((image_size, image_size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_transform = transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                                             transforms.CenterCrop((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset = Butterfly200(root=args['data_root'], is_train=True, transform=train_transform)
        testset = Butterfly200(root=args['data_root'], is_train=False, transform=test_transform)

    if args['local_rank'] == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args['local_rank'] == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args['local_rank'] == -1 else DistributedSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args['train_batch_size'],
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args['eval_batch_size'],
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
