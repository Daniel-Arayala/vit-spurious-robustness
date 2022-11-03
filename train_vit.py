import logging
import os

import numpy as np
import timm
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.comm_utils import set_seed, AverageMeter, accuracy_func
from utils.data_utils import get_loader_train
from utils.scheduler import WarmupCosineSchedule
from evaluation_utils.performance_metrics import get_classification_metrics, accuracy_score, log_evaluation

logger = logging.getLogger(__name__)

model_dict = {
    'ViT-B_16': 'vit_base_patch16_224_in21k',
    'ViT-S_16': 'vit_small_patch16_224_in21k',
    'ViT-Ti_16': 'vit_tiny_patch16_224_in21k'}


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint_dir = os.path.join(args.output_dir, args.name, args.dataset, args.model_arch)
    checkpoint_path = os.path.join(model_checkpoint_dir, args.model_type + ".bin")
    if not os.path.exists(checkpoint_path):
        os.makedirs(model_checkpoint_dir, exist_ok=True)
    torch.save(model_to_save.state_dict(), checkpoint_path)
    logger.info("Saved model checkpoint")


def setup(args):
    num_classes = args.num_classes
    model_name = model_dict[args.model_type]
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.1,
        img_size=args.img_size
    )
    model.reset_classifier(num_classes)
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_labels = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y, _ = batch
        with torch.no_grad():
            logits = model(x)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_labels[0] = np.append(
                all_labels[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_labels = all_preds[0], all_labels[0]
    val_metrics = get_classification_metrics(all_preds, all_labels)
    log_evaluation(global_step, val_metrics, writer, 'val')
    writer.add_scalar("val/loss", scalar_value=eval_losses.avg, global_step=global_step)
    writer.add_scalar("loss/val", scalar_value=eval_losses.avg, global_step=global_step)
    accuracy = accuracy_score(all_preds, all_labels)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("accuracy/val", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train_model(args):
    logger.info(f"Fine-tuning {args.model_type} on {args.dataset}")
    args, model = setup(args)
    log_dir = os.path.join("logs", args.name, args.dataset, args.model_arch, args.model_type)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    args.train_batch_size = args.train_batch_size // args.batch_split
    train_loader, val_loader = get_loader_train(args)
    cri = torch.nn.CrossEntropyLoss().to(args.device)
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.batch_split)
    logger.info("  Gradient Accumulation steps = %d", args.batch_split)

    model.zero_grad()
    set_seed(args)  
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y, _ = batch
            logits = model(x)
            loss = cri(logits.view(-1, args.num_classes), y.view(-1))
            if args.batch_split > 1:
                loss = loss / args.batch_split

            loss.backward()

            if (step + 1) % args.batch_split == 0:
                losses.update(loss.item()*args.batch_split)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                # Calculates train accuracy through iterations (every batch_split epochs)
                preds = torch.argmax(logits, dim=-1)
                preds = preds.cpu().numpy()
                labels = y.cpu().numpy()
                # train_acc = accuracy_score(labels, preds)
                train_metrics = get_classification_metrics(preds, labels)
                log_evaluation(global_step, train_metrics, writer, 'train')
                # writer.add_scalar("accuracy/train", scalar_value=train_acc, global_step=global_step)
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("loss/train", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                if global_step % args.eval_every == 0:
                    accuracy = valid(args, model, writer, val_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break
    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
