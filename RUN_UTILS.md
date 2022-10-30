# Commands for EyePacs 

## Training

```sh
    python train.py --name reduced_eyepacs --model_arch ViT --model_type ViT-Ti_16 --dataset eyepacs --warmup_steps 500 --num_steps 1000 --num_workers 2 --learning_rate 0.03 --img_size 256 --batch_split 1 --train_batch_size 64 --eval_batch_size 16 --num_classes 5
```

## Evaluation

```sh
    python evaluate.py --name reduced_eyepacs --dataset eyepacs --model_arch ViT --model_type ViT-Ti_16 --img_size 256 --batch_size 32 --num_workers 2 --num_classes 5
```

