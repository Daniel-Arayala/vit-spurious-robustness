import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import logging
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def tensor_to_list(tensor):
    return [x.item() for x in tensor]


def get_binary_results(multiclass_results):
    return [0 if x <= 1 else 1 for x in multiclass_results]


def get_classification_metrics(y_pred, y_true):
    # Convert tensor to list
    y_pred = tensor_to_list(y_pred)
    y_true = tensor_to_list(y_true)

    # Get binary results
    y_pred_bin = get_binary_results(y_pred)
    y_true_bin = get_binary_results(y_true)

    metrics = {}
    # Multiclass
    metrics["mult"] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_pred, y_true, average="macro", zero_division=0),
        "recall": recall_score(y_pred, y_true, average="macro", zero_division=0),
        "f1_score": f1_score(y_pred, y_true, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_pred, y_true, weights="quadratic"),
    }

    # Binary
    tn, fp, fn, tp = confusion_matrix(y_pred_bin, y_true_bin).ravel()
    metrics["bin"] = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_pred_bin, y_true_bin, zero_division=0),
        "recall": recall_score(y_pred_bin, y_true_bin, zero_division=0),
        "f1_score": f1_score(y_pred_bin, y_true_bin, zero_division=0),
        "roc_auc_score": roc_auc_score(y_true_bin, y_pred_bin),
        "sensitivity": (tp / (tp + fp)),
        "specificity": (tn / (tn + fp)),
    }

    return metrics


def get_metrics_binary(y_pred, y_true):
    # Convert tensor to list
    y_pred = tensor_to_list(y_pred)
    y_true = tensor_to_list(y_true)

    metrics = {}

    # Binary
    # print(confusion_matrix(y_pred, y_true).ravel())
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_pred, y_true, zero_division=0),
        "recall": recall_score(y_pred, y_true, zero_division=0),
        "f1_score": f1_score(y_pred, y_true, zero_division=0),
        # "roc_auc_score": roc_auc_score(y_true, y_pred),
        "sensitivity": (tp / (tp + fp)),
        "specificity": (tn / (tn + fp)),
    }

    try:
        metrics["roc_auc_score"] = roc_auc_score(y_true, y_pred)
    except:
        metrics["roc_auc_score"] = 0

    return metrics


def log_evaluation(epoch, statistics, writer, partition):
    if 'bin' in statistics.keys():
        # Logging all metrics for specific partition
        # Binary Classification Statistics
        writer.add_scalars(f'bin_{partition}', statistics["bin"], epoch)
        # Multiclass Classification Statistics
        writer.add_scalars(f'mult_{partition}', statistics["mult"], epoch)

        # Logging specific metric for all partitions
        for metric in statistics['mult'].keys():
            writer.add_scalar(f'{metric}/{partition}', statistics['mult'][metric], epoch)
        #logger.debug(f"{partition} epoch {epoch}. Loss {statistics['mult']['loss']} Kappa {statistics['mult']['kappa']}")
    else:
        for metric in statistics.keys():
            writer.add_scalar(f'{metric}/{partition}', statistics[metric], epoch)
            writer.add_scalar(f'{partition}/{metric}', statistics[metric], epoch)

        # logger.debug(
        #     '{} epoch {}. Loss {} Kappa {}'.format(
        #         partition, epoch, statistics['loss'], statistics['kappa']
        #     )
        # )


def log_evaluation_binary(epoch, statistics, writer, prefix):
    for metric in statistics.keys():
        writer.add_scalar("{}/{}".format(metric, prefix), statistics[metric], epoch)
        writer.add_scalar("{}/{}".format(prefix, metric), statistics[metric], epoch)


def plot_confusion_matrix(y_pred, y_true, fold, binary, args):
    cm = sns.heatmap(
        confusion_matrix(y_pred, y_true, normalize="true"),
        annot=True,
        center=0,
        vmin=0,
        vmax=1,
    )
    plt.yticks(rotation=0)
    if binary:
        img_name = "Confusion Matrix - {} - Binary".format(fold)
        plt.title(img_name)
    else:
        img_name = "Confusion Matrix - {}".format(fold)
        plt.title(img_name)

    try:
        plt.savefig(os.path.join(args.logdir, img_name))
        plt.cla()
        plt.clf()
        plt.close("all")
    except:
        logger.error(cm)
