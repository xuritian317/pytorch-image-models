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
    global testset

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':

        image_size = args.img_size
        resize_size = 600
        if image_size == 448:
            resize_size = 600
        elif image_size == 384:
            resize_size = 500
        elif image_size == 224:
            resize_size = 300

        train_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.RandomErasing()
            # transforms.RandAugment()
        ])

        test_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'butterfly200':

        image_size = args.img_size
        resize_size = 600 if args.img_size == 448 else 500

        train_transform = transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                                              transforms.RandomCrop((image_size, image_size)),
                                              transforms.RandomHorizontalFlip(),  # 水平翻转
                                              transforms.RandomVerticalFlip(),  # 上下翻转
                                              transforms.RandomRotation(15),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_transform = transforms.Compose([transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                                             transforms.CenterCrop((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset = Butterfly200(root=args.data_root, is_train=True, transform=train_transform)
        testset = Butterfly200(root=args.data_root, is_train=False, transform=test_transform)

    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root, 'devkit/cars_train_annos.mat'),
                               os.path.join(args.data_root, 'cars_train'),
                               os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                               # cleaned=os.path.join(data_dir,'cleaned.dat'),
                               transform=transforms.Compose([
                                   transforms.Resize((600, 600), Image.BILINEAR),
                                   transforms.RandomCrop((448, 448)),
                                   transforms.RandomHorizontalFlip(),
                                   AutoAugImageNetPolicy(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                               )
        testset = CarsDataset(os.path.join(args.data_root, 'cars_test_annos_withlabels.mat'),
                              os.path.join(args.data_root, 'cars_test'),
                              os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                              # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                              transform=transforms.Compose([
                                  transforms.Resize((600, 600), Image.BILINEAR),
                                  transforms.CenterCrop((448, 448)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                              )
    elif args.dataset == 'dog':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                        train=True,
                        cropped=False,
                        transform=train_transform,
                        download=False
                        )
        testset = dogs(root=args.data_root,
                       train=False,
                       cropped=False,
                       transform=test_transform,
                       download=False
                       )
    elif args.dataset == 'nabirds':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                              transforms.RandomCrop((304, 304)),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                             transforms.CenterCrop((304, 304)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
