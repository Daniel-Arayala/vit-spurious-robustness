import argparse

import torch
from utils.data_utils import get_loader_train


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--name", required=True, help="Name of this run. Used for monitoring."
    )
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
        "-lr",
        "--learning_rate",
        default=3e-2,
        type=float,
        help="The initial learning rate for SGD.",
    )
    parser.add_argument(
        "--weight_decay", default=0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--num_steps",
        default=1500,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
        help="Step of training to perform learning rate warmup for.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--metric_types", choices=["bin", "bin_out", "mult"], default=["bin", "mult"], nargs='+',
        help="Types of metrics to calculate. If 'bin' indicates that the problem is binary by nature. "
             "If 'bin_out' the multiclass output will be converted to binary and the metrics calculated. "
             "If 'mult' the multiclass metrics will be calculated."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--batch_split",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        "--use_clearml",
        action="store_true",
        help="Whether to use the ClearML tool as an experiment tracker",
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

    args = parser.parse_args()

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def main():
    args = parse_args()


if __name__ == '__main__':
    train_loader = get_loader_train()
    get_mean_and_std()
