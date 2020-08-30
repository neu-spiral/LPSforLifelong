export CUDA_VISIBLE_DEVICES=0
#!/bin/bash
exp='test'
echo $exp
echo "Start JOB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
savepath='/'

python -u main.py \
    --exp_name $exp \
    --base_path /scratch/RTML/lifelong/cifar/ \
    --save_path $savepath \
    --dataset cifar \
    --tasks 6 \
       --load-model '' \
       --load-model-pruned '' \
       --arch cifarnet \
       --input_size 32 \
       --classes 10 \
       --batch-size 128 \
       --multi-gpu \
       --no-tricks \
       --sparsity-type irregular \
       --epochs 200 \
       --epochs-prune 200 \
       --epochs-mask-retrain 200 \
       --admm-epochs 3 \
       --mask-admm-epochs 10 \
       --optmzr adam \
       --rho 0.01 \
       --rho-num 3 \
       --lr 0.0005 \
       --lr-scheduler cosine \
       --warmup \
       --warmup-epochs 5 \
       --mixup \
       --alpha 0.3 \
       --smooth \
       --smooth-eps 0.1 \
       --config-setting 5,1,1,1,1,1 \
       --adaptive-mask True \
       --adaptive-ratio 0.9 \
       >$savepath/$exp/log.out &&
echo "Congratus! Finished admm training!"
