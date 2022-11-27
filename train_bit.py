import logging
import os
import time
import urllib.request as url_request
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models.bits as models
import utils.bit_common as bit_common
import utils.bit_hyperrule as bit_hyperrule
import utils.lbtoolbox as lb
from evaluation_utils.performance_metrics import (
    get_classification_metrics,
    log_evaluation,
)
from utils.comm_utils import AverageMeter
from utils.data_utils import get_loader_train

BIT_STORAGE_TEMPLATE_URL = (
    "https://storage.googleapis.com/bit_models/%(model_type)s.npz"
)

logger = logging.getLogger(__name__)


def accuracy(out, label):
    _, pred = torch.max(out, dim=1)
    return torch.tensor(torch.sum(pred == label).item() / len(pred))


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def run_eval(model, data_loader, writer, device, chrono, step):
    model.eval()
    logger.info("Running validation...")
    eval_losses = AverageMeter()
    all_preds, all_labels, all_probs = [], [], []
    epoch_iterator = tqdm(
        data_loader,
        desc="Validating (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        # file=TqdmToLogger(logger, level=logging.INFO),
        dynamic_ncols=True,
    )
    end = time.time()
    for x, y, g in epoch_iterator:
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            chrono._done("eval load", time.time() - end)
            with chrono.measure("eval fprop"):
                logits = model(x)
                eval_loss = torch.nn.CrossEntropyLoss()(logits, y)
                eval_losses.update(eval_loss.item())
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())
                all_probs.extend(F.softmax(logits).cpu().detach().numpy())

        epoch_iterator.set_description("Validating (loss=%2.5f)" % eval_losses.val)
        end = time.time()
    val_metrics = get_classification_metrics(all_labels, all_preds, all_probs)
    log_evaluation(step, val_metrics, writer, "val")
    writer.add_scalar("loss/val", scalar_value=eval_losses.avg, global_step=step)
    try:
        accuracy = val_metrics["mult"]["accuracy"]
    except KeyError:
        accuracy = val_metrics["bin"]["accuracy"]

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def load_bit_model(model, model_type, logger):
    bit_pretrained_dir = "bit_pretrained_models"
    os.makedirs(bit_pretrained_dir, exist_ok=True)
    full_path = os.path.join(bit_pretrained_dir, model_type + ".npz")
    bit_url = BIT_STORAGE_TEMPLATE_URL % {"model_type": model_type}
    if not os.path.exists(full_path):
        logger.info("Downloading BiT model from repository")
        _ = url_request.urlretrieve(bit_url, full_path)
        logger.info(f"Downloaded BiT Model type: {model_type}")
    logger.info("Loading BiT model...")
    model.load_from(np.load(full_path))
    logger.info(f"BiT Model Loaded: {model_type}")


def train_model(args):
    logger = bit_common.setup_logger(args)
    logger.info(f"Fine-tuning {args.model_type} on {args.dataset}")
    torch.backends.cudnn.benchmark = True
    logger.info(f"Training on {args.device}")
    log_dir = os.path.join(
        "logs", args.name, args.dataset, args.model_arch, args.model_type
    )
    writer = SummaryWriter(log_dir=log_dir)
    args.train_batch_size = args.train_batch_size // args.batch_split
    train_loader, valid_loader = get_loader_train(args)

    logger.info(f"Loading model from {args.model_type}.npz")
    model = models.KNOWN_MODELS[args.model_type](
        head_size=args.num_classes, zero_head=True
    )
    load_bit_model(model, args.model_type, logger)
    logger.info("Moving model to GPU")
    model = torch.nn.DataParallel(model)
    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    model_checkpoint_dir = pjoin(
        args.output_dir, args.name, args.dataset, args.model_arch
    )
    savename = pjoin(model_checkpoint_dir, args.model_type + ".pth.tar")
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir, exist_ok=True)

    model: callable = model.to(args.device)
    optim.zero_grad()

    model.train()
    cri: callable = torch.nn.CrossEntropyLoss().to(args.device)

    logger.info("Starting training!")
    epoch_iterator = tqdm(
        train_loader,
        desc="Training (X / X Steps) (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        # file=TqdmToLogger(logger, level=logging.INFO),
        dynamic_ncols=True,
    )
    chrono = lb.Chrono()
    end = time.time()
    global_step, best_acc = 0, 0
    t_total = args.num_steps
    # Accumulates the predictions and labels for all batch splits adding to one full batch size
    preds_effective_batch, labels_effective_batch, probs_effective_batch = [], [], []
    batch_loss_accum = 0  # Accumulates the average loss per split inside a batch
    with lb.Uninterrupt() as u:
        while True:
            for step, batch in enumerate(epoch_iterator):
                x, y, _ = batch
                chrono._done("load", time.time() - end)
                if u.interrupted:
                    break
                # Schedule sending to GPU(s)
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                # Update learning-rate, including stop training if over.
                lr = bit_hyperrule.get_lr(
                    global_step, len(train_loader.dataset), base_lr=args.learning_rate
                )
                if lr is None:
                    logging.warning("Learning rate is None. Exiting training function")
                    break
                for param_group in optim.param_groups:
                    param_group["lr"] = lr
                # compute output
                with chrono.measure("fprop"):
                    logits = model(x)
                    loss = cri(logits.view(-1, args.num_classes), y.view(-1))
                    # Accumulate grads
                    with chrono.measure("grads"):
                        loss = loss / args.batch_split
                        batch_loss_accum += loss.item()
                        loss.backward()
                    # Batch split metrics
                    preds = torch.argmax(logits, dim=-1)
                    preds = preds.cpu().numpy()
                    labels = y.cpu().numpy()
                    probs = F.softmax(logits).cpu().detach().numpy()
                    # Accumulating true and predicted for the whole batch size
                    preds_effective_batch.extend(preds)
                    labels_effective_batch.extend(labels)
                    probs_effective_batch.extend(probs)

                # accstep = f"({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
                # logger.info(f"[Accum steps: {accstep}]: loss={loss_num:.5f} (lr={lr:.1e})")
                # logger.flush()

                # Update params
                if ((step + 1) % args.batch_split == 0) or (
                    step + 1 == len(epoch_iterator)
                ):
                    with chrono.measure("update"):
                        optim.step()
                        optim.zero_grad()
                    # Calculates train accuracy through iterations (every batch_split epochs)
                    train_metrics = get_classification_metrics(
                        preds_effective_batch,
                        labels_effective_batch,
                        probs_effective_batch,
                    )
                    # Updating variables (Do not change position os these 2 lines)
                    global_step += 1
                    # Logging metrics
                    log_evaluation(global_step, train_metrics, writer, "train")
                    # writer.add_scalar("accuracy/train", scalar_value=train_acc, global_step=global_step)
                    writer.add_scalar(
                        "loss/train",
                        scalar_value=batch_loss_accum,
                        global_step=global_step,
                    )
                    writer.add_scalar("lr", scalar_value=lr, global_step=global_step)
                    # Setting accumulators to empty to receive the next batch
                    (
                        preds_effective_batch,
                        labels_effective_batch,
                        probs_effective_batch,
                    ) = ([], [], [])
                    epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)"
                        % (global_step, t_total, batch_loss_accum)
                    )
                    batch_loss_accum = 0
                    if args.eval_every and global_step % args.eval_every == 0:
                        acc = run_eval(
                            model,
                            valid_loader,
                            writer,
                            args.device,
                            chrono,
                            global_step,
                        )
                        if best_acc < acc:
                            logger.info("Saved model checkpoint")
                            best_acc = acc
                            torch.save(
                                {
                                    "step": global_step,
                                    "model": model.state_dict(),
                                    "optim": optim.state_dict(),
                                },
                                savename,
                            )
                        model.train()

                    if global_step % t_total == 0:
                        break
                end = time.time()

            if (global_step % t_total == 0) or (lr is None):
                break
    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    logger.info(f"Timings:\n{chrono}")
