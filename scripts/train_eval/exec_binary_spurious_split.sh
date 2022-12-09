# ViT Training
python train.py --name spurious_binary_split_full_eyepacs --model_arch ViT --model_type ViT-S_16 --metadata_file_name metadata_binary_spurious --dataset eyepacs --dataset_folder_name eyepacs --metric_types bin --warmup_steps 200 --num_steps 500 --num_workers 6 --learning_rate 0.03 --img_size 256 --batch_split 32 --train_batch_size 512 --eval_batch_size 16 --eval_every 10 --num_classes 2 --use_clearml
# ViT Evaluation
python evaluate.py --name spurious_binary_split_full_eyepacs --model_arch ViT --model_type ViT-S_16 --metadata_file_name metadata_binary_spurious --dataset eyepacs --dataset_folder_name eyepacs --metric_types bin --num_workers 6 --img_size 256 --batch_size 32 --num_classes 2 --save_prediction_info --use_clearml

# BiT Training
python train.py --name spurious_binary_split_full_eyepacs --model_arch BiT --model_type BiT-M-R50x1 --metadata_file_name metadata_binary_spurious --dataset eyepacs --dataset_folder_name eyepacs --metric_types bin --warmup_steps 200 --num_steps 500 --num_workers 6 --learning_rate 0.003 --img_size 256 --batch_split 32 --train_batch_size 512 --eval_batch_size 16 --eval_every 10 --num_classes 2 --use_clearml
# BiT Evaluation
python evaluate.py --name spurious_binary_split_full_eyepacs --model_arch BiT --model_type BiT-M-R50x1 --metadata_file_name metadata_binary_spurious --dataset eyepacs --dataset_folder_name eyepacs --metric_types bin --num_workers 6 --img_size 256 --batch_size 32 --num_classes 2 --save_prediction_info --use_clearml