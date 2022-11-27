import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import timm
import torch
from clearml import Logger
from tqdm import tqdm

import models.bits as bits
from constants import KNOWN_MODELS
from performance_metrics import get_classification_metrics, log_metrics_to_clearml
from utils.data_utils import get_loader_inference, get_loader_train

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self):
        # Contains prediction, label and environment
        self.prediction_info = defaultdict(list)
        self.df_prediction_info = None

    def _consolidate_batch_info(self):
        self.df_prediction_info = pd.DataFrame(self.prediction_info)

    @staticmethod
    def _convert_to_correct_output_fmt(metric_info, include_cm, output_fmt):
        if include_cm and output_fmt == 'df':
            metrics, cm = metric_info
            return pd.DataFrame(metrics).T, cm
        elif output_fmt == 'df':
            return pd.DataFrame(metric_info).T
        elif output_fmt == 'dict':
            return metric_info
        else:
            raise ValueError('Invalid arguments passed to the method.')

    @staticmethod
    def _adjust_output_metric_info(metric_info, include_cm, output_fmt):
        if isinstance(metric_info, dict):
            for class_type, class_type_metric_info in metric_info.items():
                metric_info[class_type] = Metrics._convert_to_correct_output_fmt(
                    class_type_metric_info, include_cm, output_fmt)
            return metric_info
        else:
            return Metrics._convert_to_correct_output_fmt(metric_info, include_cm, output_fmt)

    def append_batch_info(self, labels, preds, probs, envs=None):
        self.prediction_info['labels'].extend(labels)
        self.prediction_info['preds'].extend(preds)
        self.prediction_info['probs'].extend(probs)
        if envs is not None:
            self.prediction_info['envs'].extend(envs)

    def calculate_metrics(self, include_cm=False, output_fmt='dict'):
        if self.df_prediction_info is None:
            self._consolidate_batch_info()
        metric_info = get_classification_metrics(
            y_true=self.df_prediction_info['labels'].values,
            y_pred=self.df_prediction_info['preds'].values,
            probs=self.df_prediction_info['probs'].values,
            include_cm=include_cm)
        return Metrics._adjust_output_metric_info(metric_info, include_cm, output_fmt)

    def calculate_metrics_per_env(self, include_cm=False, output_fmt='dict'):
        if self.df_prediction_info is None:
            self._consolidate_batch_info()
        env_metric_info = self.df_prediction_info.groupby('envs').apply(
            lambda df_env: get_classification_metrics(
                y_true=df_env['labels'].values,
                y_pred=df_env['preds'].values,
                probs=df_env['probs'].values,
                include_cm=include_cm))
        adjusted_env_metrics = {
            env_id: Metrics._adjust_output_metric_info(metrics, include_cm, output_fmt)
            for env_id, metrics in env_metric_info.to_dict().items()}
        return adjusted_env_metrics

    def count_samples_per_env(self):
        if self.df_prediction_info is None:
            self._consolidate_batch_info()
        df_env_count: pd.Series = self.df_prediction_info.groupby('envs')['preds'].count()
        df_env_count.name = 'count'
        return df_env_count


def get_partition_metrics(loader, model, partition):
    metrics = Metrics()
    count = 0
    epoch_iterator = tqdm(loader,
                          desc=f'Evaluating metrics for {partition} partition',
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for data in epoch_iterator:
        images, labels, envs = data
        count += len(images)
        inputs = images.cuda()
        inputs.requires_grad = True
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1)
        metrics.append_batch_info(
            labels=labels.cpu().numpy(),
            preds=preds.cpu().numpy(),
            probs=logits.cpu().detach().numpy(),
            envs=envs.cpu().numpy())
        # if count > 16: break

    return {
        'partition': metrics.calculate_metrics(include_cm=True, output_fmt='df'),
        'env': metrics.calculate_metrics_per_env(include_cm=True, output_fmt='df')
    }


def calculate_inference_metrics(args):
    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(
            args.output_dir, args.name, args.dataset, args.model_arch, args.model_type)

    if args.model_arch == "ViT" or args.model_arch == "DeiT":
        model = timm.create_model(
            KNOWN_MODELS[args.model_type],
            pretrained=False,
            num_classes=args.num_classes,
            img_size=args.img_size
        )
        model.load_state_dict(torch.load(args.checkpoint_dir + ".bin"))
        model.eval()
    elif args.model_arch == "BiT":
        model = bits.KNOWN_MODELS[args.model_type](head_size=args.num_classes, zero_head=False)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(args.checkpoint_dir + ".pth.tar", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    else:
        raise NotImplementedError(f'Invalid architecture name: {args.model_arch}')

    try:
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception:
        raise Exception("No CUDA enabled device found. Please Check!")

    logger.info(f"Inference for Dataset: {args.dataset} | Model : {args.model_type} ")
    train_loader, val_loader, test_loader = get_loader_inference(args, include_val=True)
    # Train
    logger.info("Calculating Metrics for the Train data")
    result_train: dict = get_partition_metrics(train_loader, model, partition='train')
    logger.info(f'Train Metrics: {result_train}')

    # Validation
    logger.info("Calculating Metrics for the Validation data")
    result_val: dict = get_partition_metrics(val_loader, model, partition='val')
    logger.info(f'Test Metrics: {result_val}')
    # Test
    logger.info("Calculating Metrics for the Test data")
    result_test: dict = get_partition_metrics(test_loader, model, partition='test')
    logger.info(f'Test Metrics: {result_test}')

    if args.use_clearml:
        clearml_logger = Logger.current_logger()
        logger.info('Adding train metrics to ClearML')
        log_metrics_to_clearml(result_train, 'train', clearml_logger)
        logger.info('Adding validation metrics to ClearML')
        log_metrics_to_clearml(result_val, 'validation', clearml_logger)
        logger.info('Adding test metrics to ClearML')
        log_metrics_to_clearml(result_test, 'test', clearml_logger)


def main():
    num_random_samples = 3
    sample_size = 20
    metrics = Metrics()
    for i in range(num_random_samples):
        sample = np.hstack((np.random.randint(0, 5, (sample_size, 2)),  # True and predicted (Grade 0 to 4)
                            np.random.randint(0, 4, (sample_size, 1))))  # Environment (0 to 3)
        labels, preds, envs = sample[:, 0], sample[:, 1], sample[:, 2]
        metrics.append_batch_info(labels, preds, envs)
    metrics_all = metrics.calculate_metrics(include_cm=True, output_fmt='df')
    metrics_per_env = metrics.calculate_metrics_per_env(include_cm=True, output_fmt='df')
    sample_count_per_env = metrics.count_samples_per_env()


if __name__ == '__main__':
    main()
