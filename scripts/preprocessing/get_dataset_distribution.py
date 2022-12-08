import argparse

import torch
from utils.data_utils import get_loader_train, get_mean_and_std


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dataset",
        choices=["waterbirds", "cmnist", "celebA", "eyepacs"],
        default="waterbirds",
        help="Which downstream task.",
    )
    parser.add_argument(
        "--model_arch",
        choices=["ViT", "BiT"],
        default="ViT",
        help="Which variant to use.",
    )
    parser.add_argument(
        "--model_type", default="ViT-B_16", help="Which variant to use."
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where checkpoints will be written.",
    )
    parser.add_argument("--img_size", default=384, type=int, help="Resolution size")
    parser.add_argument(
        "--train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--eval_every",
        default=100,
        type=int,
        help="Run prediction on validation set every so many steps."
             "Will always run one evaluation at the end of training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers to use for the DataLoader.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for the classification task.",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata",
        help="Name of the metadata csv file containing the image metadata and splits. Defaults to 'metadata'."
    )
    parser.add_argument(
        "--dataset_folder_name",
        type=str,
        default="reduced_eyepacs_resized_cropped",
        help="Name of the folder where thedataset is located. Defaults to 'reduced_eyepacs_resized_cropped'"
    )
    parser.add_argument(
        "--remove_crop_resize",
        action="store_true",
        help="Whether to remove the random crops and resizing of the input images.",
    )
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Handle imbalanced classes with WeightedRandomSampler",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Use data augmentation based on kaggle competition notebooks",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_loader, _ = get_loader_train(args)
    mean, std = get_mean_and_std(train_loader)
    print(f'Mean: {mean} | Std: {std}')


if __name__ == '__main__':
    main()
