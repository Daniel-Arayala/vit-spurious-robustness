import argparse
import logging

from evaluation_utils.evaluate_acc import calculate_acc

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="help identify checkpoint")
    parser.add_argument("--dataset", choices=["waterbirds", "cmnist", "celebA", "eyepacs"], default="waterbirds",
                        help="Which downstream task.")
    parser.add_argument("--model_arch", choices=["ViT", "BiT"],
                        default="ViT",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint_dir",
                        help="directory of saved model checkpoint")
    parser.add_argument("--model_type", default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The directory where checkpoints are stored.")
    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of workers to use for the DataLoader.")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of classes for the classification task.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    calculate_acc(args)


if __name__ == "__main__":
    main()
