#! /bin/sh

python evaluate.py --name vanilla --dataset eyepacs --model_arch ViT --model_type ViT-S_16 --img_size 256 --batch_size 512 --num_workers 2 --num_classes 5 --use_clearml
python evaluate.py --name vanilla --dataset eyepacs --model_arch BiT --model_type BiT-M-R50x1 --img_size 256 --batch_size 512 --num_workers 2 --num_classes 5 --use_clearml

python evaluate.py --name augmentation --dataset eyepacs --model_arch ViT --model_type ViT-S_16 --img_size 256 --batch_size 512 --num_workers 2 --num_classes 5 --use_clearml
python evaluate.py --name augmentation --dataset eyepacs --model_arch BiT --model_type BiT-M-R50x1 --img_size 256 --batch_size 512 --num_workers 2 --num_classes 5 --use_clearml
