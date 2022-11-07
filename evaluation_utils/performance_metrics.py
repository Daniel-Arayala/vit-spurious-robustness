import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

from datasets.constants import CLASS_TYPE_MAP, EYEPACS_ENV_MAP_INV

logger = logging.getLogger(__name__)


def tensor_to_list(tensor):
    return [x.item() for x in tensor]


def get_binary_results(multiclass_results):
    return [0 if x <= 1 else 1 for x in multiclass_results]


def convert_inputs_to_list(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = tensor_to_list(y_true)
    if isinstance(y_pred, torch.Tensor):
        y_pred = tensor_to_list(y_pred)
    return y_true, y_pred


def get_metrics(y_true, y_pred, class_type='bin', include_cm=False):
    # Convert tensor to list
    y_true, y_pred = convert_inputs_to_list(y_true, y_pred)

    labels = list(range(5))
    # Binarization
    if class_type == 'bin':
        y_true = get_binary_results(y_true)
        y_pred = get_binary_results(y_pred)
        labels = [0, 1]

    metrics = classification_report(y_true, y_pred, output_dict=True)

    if include_cm:
        return metrics, confusion_matrix(y_true, y_pred, labels=labels)
    return metrics


def log_evaluation(epoch, statistics, writer, partition, metric_scope='global'):
    """metric_scope:
        'class': logs precision, recall, and f1-score for each class separately
        'global': logs precision, recall, and f1-score across all classes (macro avg, weighted avg, and micro avg)
        'all': logs both class-specific metrics and global metrics
    """
    statistics_keys = list(statistics.keys())
    has_classification_type = ('bin' in statistics_keys) or ('mult' in statistics_keys)
    if has_classification_type:
        for classification_type, metric_info in statistics.items():
            cl_type = CLASS_TYPE_MAP[classification_type]
            for metric_name, metric_value in metric_info.items():
                # Drops the support field from the classification report
                if isinstance(metric_value, dict):
                    _ = metric_value.pop('support', None)
                if metric_name.isnumeric() and metric_scope in ('class', 'all'):
                    graph_title = f'{partition.title()} - {cl_type} - Class {metric_name}'
                    writer.add_scalars(graph_title, metric_value, epoch)
                elif metric_name == 'accuracy' and metric_scope in ('global', 'all'):
                    writer.add_scalar(f'{metric_name.title()}/{partition}', metric_value, epoch)
                # macro avg, weighted avg, and micro avg
                elif (not metric_name.isnumeric()) and (metric_scope in ('global', 'all')):
                    # Grouping by metric and partition
                    for nested_metric_name, nested_metric_value in metric_value.items():
                        graph_title_part_group = \
                            f'{partition.title()} - {cl_type} - {metric_name.title()}/{nested_metric_name}'
                        writer.add_scalar(graph_title_part_group, nested_metric_value, epoch)
                        graph_title_metric_group = \
                            f'{cl_type} - {nested_metric_name.title()} ({metric_name.title()})/{partition}'
                        writer.add_scalar(graph_title_metric_group, nested_metric_value, epoch)


def log_metrics_to_clearml(metrics, partition, clearml_logger):
    base_output_dir = os.path.join(Path(__file__).parents[1], 'metrics')
    os.makedirs(base_output_dir, exist_ok=True)
    # Metrics for the entire train, validation, or test partitions
    part_metrics_bin = metrics['partition']['bin']
    part_metrics_mult = metrics['partition']['mult']
    # Binary
    if isinstance(part_metrics_bin, tuple):
        part_metrics_bin, cm_bin = part_metrics_bin
        plot_confusion_matrix(
            cm=cm_bin,
            title=f'Confusion Matrix {partition.title()} (Binary)',
            xlabel='Predicted', ylabel='Actual',
            output_path=os.path.join(base_output_dir, f'CMPart{partition.title()}Bin.png'))
    clearml_logger.report_table(
        title=f'{partition.title()} Metrics (Binary)',
        series='', table_plot=part_metrics_bin)

    if isinstance(part_metrics_mult, tuple):
        part_metrics_mult, cm_mult = part_metrics_mult
        plot_confusion_matrix(
            cm=cm_mult,
            title=f'Confusion Matrix {partition.title()} (Multiclass)',
            xlabel='Predicted', ylabel='Actual',
            output_path=os.path.join(base_output_dir, f'CMPart{partition.title()}Mult.png'))
    clearml_logger.report_table(
        title=f'{partition.title()} Metrics (Multiclass)',
        series='', table_plot=part_metrics_mult)

    # Metrics for Each environment
    for env_id, env_metrics in metrics['env'].items():
        env_metrics_bin = env_metrics['bin']
        env_metrics_mult = env_metrics['mult']

        # Binary
        if isinstance(env_metrics_bin, tuple):
            env_metrics_bin, env_cm_bin = env_metrics_bin
            plot_confusion_matrix(
                cm=env_cm_bin,
                title=f'Confusion Matrix {partition.title()} - Environment: {EYEPACS_ENV_MAP_INV[env_id]} (Binary)',
                xlabel='Predicted', ylabel='Actual',
                output_path=os.path.join(base_output_dir, f'CMEnv{env_id}{partition.title()}Bin.png'))
        clearml_logger.report_table(
            title=f'{partition.title()} Metrics - Environment: {EYEPACS_ENV_MAP_INV[env_id]} (Binary)',
            series='', table_plot=env_metrics_bin)

        if isinstance(env_metrics_mult, tuple):
            env_metrics_mult, env_cm_mult = env_metrics_mult
            plot_confusion_matrix(
                cm=env_cm_mult,
                title=f'Confusion Matrix {partition.title()} - Environment: {EYEPACS_ENV_MAP_INV[env_id]} (Multiclass)',
                xlabel='Predicted', ylabel='Actual',
                output_path=os.path.join(base_output_dir, f'CMEnv{env_id}{partition.title()}Mult.png'))
        clearml_logger.report_table(
            title=f'{partition.title()} Metrics - Environment: {EYEPACS_ENV_MAP_INV[env_id]} (Multiclass)',
            series='', table_plot=env_metrics_mult)


def get_classification_metrics(y_true, y_pred, include_cm=False, class_types=('bin', 'mult')):
    return {
        class_type: get_metrics(y_true, y_pred, class_type=class_type, include_cm=include_cm)
        for class_type in class_types}


def log_evaluation_binary(epoch, statistics, writer, prefix):
    for metric in statistics.keys():
        writer.add_scalar("{}/{}".format(metric, prefix), statistics[metric], epoch)
        writer.add_scalar("{}/{}".format(prefix, metric), statistics[metric], epoch)


def plot_confusion_matrix_from_preds(y_pred, y_true, fold, binary, args):
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


def plot_confusion_matrix(cm, title, xlabel, ylabel, output_path, output_fmt='both'):
    """
    output_fmt --> 'pct', 'abs', 'both'
    """
    if output_fmt == 'abs':
        vmin, vmax = None, None
        fmt = 'd'
        annot = True
        cbar = True
    elif output_fmt == 'pct':
        vmin, vmax = 0, 100
        cm = cm / np.sum()
        fmt = '.2%'
        annot = True
    elif output_fmt:
        cm_pct = cm / np.sum(cm)
        vmin, vmax = 0, 100
        cm_pct_str = [f'{value:0,.2%}' for value in cm_pct.flatten()]
        cm_counts_str = [f'{value:0.0f}' for value in cm.flatten()]
        annot = [f'{pct_value}\n\n({abs_value})' for pct_value, abs_value in zip(cm_pct_str, cm_counts_str)]
        annot = np.array(annot).reshape(cm.shape)
        cm = cm_pct
        fmt = ''
    else:
        raise ValueError(f'Invalid argument for output_fmt: {output_fmt}')
    cm = sns.heatmap(
        data=cm,
        annot=annot,
        fmt=fmt,
        vmin=vmin, vmax=vmax
    )
    plt.yticks(rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    try:
        plt.savefig(output_path)
        plt.cla()
        plt.clf()
        plt.close("all")
    except:
        logger.error(cm)
