#! /bin/sh

python train.py --name vanilla --model_arch BiT --model_type BiT-M-R50x1 --dataset eyepacs --num_steps 500 --num_workers 2 --learning_rate 0.003 --img_size 256 --batch_split 32 --train_batch_size 512 --eval_batch_size 128 --eval_every 10 --num_classes 5 --use_clearml
python evaluate.py --name vanilla --dataset eyepacs --model_arch BiT --model_type BiT-M-R50x1 --img_size 256 --batch_size 512 --num_workers 2 --num_classes 5 --use_clearml

python train.py --name vanilla --model_arch ViT --model_type ViT-S_16 --dataset eyepacs --warmup_steps 200 --num_steps 500 --num_workers 2 --learning_rate 0.03 --img_size 256 --batch_split 32 --train_batch_size 512 --eval_batch_size 128 --eval_every 10 --num_classes 5 --use_clearml
python evaluate.py --name vanilla --dataset eyepacs --model_arch ViT --model_type ViT-S_16 --img_size 256 --batch_size 512 --num_workers 2 --num_classes 5 --use_clearml
