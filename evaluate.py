import argparse
import logging

from evaluation_utils.evaluate_metrics import calculate_inference_metrics
from clearml import Task, TaskTypes

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
    parser.add_argument("--dataset_folder_name", type=str,
                        default="reduced_eyepacs_resized_cropped",
                        help="Name of the folder where thedataset is located. "
                             "Defaults to 'reduced_eyepacs_resized_cropped'")
    parser.add_argument("--metadata_file_name", type=str,
                        default="metadata",
                        help="Name of the metadata csv file containing the image "
                             "metadata and splits. Defaults to 'metadata'.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--use_clearml', action='store_true',
                        help="Whether to use the ClearML tool as an experiment tracker")
    parser.add_argument('--save_prediction_info', action='store_true',
                        help="Whether to save the detailed prediction information per image.")
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    task = None
    if args.use_clearml:
        # Setting up ClearML tracking
        task = Task.init(
            project_name=f'ViTs Robustness to Spurious Correlation/{args.name}/{args.model_arch}',
            task_name=f'Evaluating {args.model_type} on {args.dataset}',
            task_type=TaskTypes.inference,
            #reuse_last_task_id=False,
            tags=[args.model_arch, args.model_type, args.dataset]
        )
    calculate_inference_metrics(args)
    logger.info('Evaluation finished!')
    if args.use_clearml:
        task.mark_completed()


if __name__ == "__main__":
    main()
