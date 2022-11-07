import logging

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from datasets.celebA_dataset import get_celebA_dataloader, get_celebA_dataset
from datasets.waterbirds_dataset import get_waterbird_dataloader, get_waterbird_dataset
from datasets.color_mnist import get_biased_mnist_dataloader, get_biased_mnist_dataset
from datasets.eyepacs_dataset import get_eyepacs_dataloader, get_eyepacs_dataset


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96 * 96 else (512, 480)


known_dataset_sizes = {

    'cmnist': (28, 28),
    'waterbirds': (224, 224),
    'celebA': (224, 224),
    'eyepacs': (1024, 1024)
}


def get_normalize_params(args):
    if args.model_arch == "DeiT":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    return mean, std


def get_resolution_from_dataset(dataset):
    if dataset not in known_dataset_sizes:
        raise ValueError(f"Unsupported dataset {dataset}.")
    return get_resolution(known_dataset_sizes[dataset])


def get_loader_train(args):
    mean, std = get_normalize_params(args)
    print(mean, std)
    if args.model_arch == "BiT":
        precrop, crop = get_resolution_from_dataset(args.dataset)
        transform_train = transforms.Compose([
            transforms.Resize((precrop, precrop)),
            transforms.RandomCrop((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_val = transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # Transformation steps
        transform_train_steps = [
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        transform_val_steps = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        if args.remove_crop_resize:
            transform_train_steps = transform_train_steps[1:]
            transform_val_steps = transform_val_steps[1:]

        transform_train = transforms.Compose(transform_train_steps)
        transform_val = transforms.Compose(transform_val_steps)

    if args.dataset == "celebA":
        train_set = get_celebA_dataset(split="train", transform=transform_train,
                                      root_dir='datasets')
        val_set = get_celebA_dataset(split="val", transform=transform_val,
                                     root_dir='datasets')
    elif args.dataset == "waterbirds":
        train_set = get_waterbird_dataset(data_label_correlation=0.95,
                                         split="train", transform=transform_val,
                                         root_dir='datasets')

        val_set = get_waterbird_dataset(data_label_correlation=0.95,
                                        split="val", transform=transform_val, root_dir='datasets')
    elif args.dataset == "cmnist":
        trainset_1 = get_biased_mnist_dataset(root='./datasets/MNIST',
                                              data_label_correlation=0.45,
                                              n_confusing_labels=1,
                                              train=True, partial=True, cmap="1", transform=transform_train)
        trainset_2 = get_biased_mnist_dataset(root='./datasets/MNIST',
                                              data_label_correlation=0.45,
                                              n_confusing_labels=1,
                                              train=True, partial=True, cmap="2", transform=transform_train)
        val_set = get_biased_mnist_dataset(root='./datasets/MNIST',
                                           data_label_correlation=0.45,
                                           n_confusing_labels=1,
                                           train=False, partial=True, cmap="1", transform=transform_val)
        train_set = trainset_1 + trainset_2

    elif args.dataset == "eyepacs":
        train_set = get_eyepacs_dataset(root_dir='datasets',
                                        dataset_name='reduced_eyepacs_resized_cropped',
                                        split='train', transform=transform_train)

        val_set = get_eyepacs_dataset(root_dir='datasets',
                                      dataset_name='reduced_eyepacs_resized_cropped',
                                      split='val', transform=transform_val)
    else:
        raise NotImplemented(f'Invalid dataset option: {args.dataset}')

    train_sampler = RandomSampler(train_set)
    val_sampler = SequentialSampler(val_set)
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True) if val_set is not None else None

    return train_loader, val_loader


def get_loader_inference(args, include_val=False):
    mean, std = get_normalize_params(args)
    val_set = None
    if args.model_arch == "BiT":
        precrop, crop = get_resolution_from_dataset(args.dataset)
        transform_test = transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if args.dataset == "celebA":
        train_set = get_celebA_dataset(split="train", transform=transform_test,
                                       root_dir='datasets')
        test_set = get_celebA_dataset(split="test", transform=transform_test,
                                      root_dir='datasets')
    elif args.dataset == "waterbirds":
        train_set = get_waterbird_dataset(data_label_correlation=0.95,
                                          split="train", transform=transform_test, root_dir='datasets')

        test_set = get_waterbird_dataset(data_label_correlation=0.95,
                                         split="test", transform=transform_test, root_dir='datasets')

    elif args.dataset == "cmnist":
        trainset_1 = get_biased_mnist_dataset(root='./datasets/MNIST',
                                              data_label_correlation=0.45,
                                              n_confusing_labels=1,
                                              train=True, partial=True, cmap="1", transform=transform_test)
        trainset_2 = get_biased_mnist_dataset(root='./datasets/MNIST',
                                              data_label_correlation=0.45,
                                              n_confusing_labels=1,
                                              train=True, partial=True, cmap="2", transform=transform_test)
        test_set = get_biased_mnist_dataset(root='./datasets/MNIST',
                                            data_label_correlation=0.45,
                                            n_confusing_labels=1,
                                            train=False, partial=True, cmap="1", transform=transform_test)
        train_set = trainset_1 + trainset_2

    elif args.dataset == "eyepacs":
        train_set = get_eyepacs_dataset(root_dir='datasets',
                                        dataset_name='reduced_eyepacs_resized_cropped',
                                        split='train', transform=transform_test)

        test_set = get_eyepacs_dataset(root_dir='datasets',
                                       dataset_name='reduced_eyepacs_resized_cropped',
                                       split='test', transform=transform_test)
        if include_val:
            val_set = get_eyepacs_dataset(root_dir='datasets',
                                          dataset_name='reduced_eyepacs_resized_cropped',
                                          split='val', transform=transform_test)

    else:
        raise NotImplemented(f'Invalid dataset option: {args.dataset}')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True) if test_set is not None else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers,
                                              pin_memory=True) if test_set is not None else None
    if include_val:
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers,
                                                 pin_memory=True) if test_set is not None else None
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader
