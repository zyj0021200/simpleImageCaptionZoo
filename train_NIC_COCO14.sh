start_from=$1
gpu_id=$2

python -u Main.py --dataset COCO14 \
    --model_type NIC \
    --gpu_id gpu_id \
    --operation train \
    --start_from start_from \
    --use_bu unused \
    --num_epochs 30 \
    --train_batch_size 128 \
    --label_smoothing 0.1 \
    --learning_rate 4e-4 \
    --cnn_finetune_learning_rate 1e-4 \
    --cnn_finetune_start 8

