#!/bin/bash

amzn_datasets=("amzn_apparel 5 7" "amzn_bath 5 7" "amzn_bedding 2 10" "amzn_carseats 1 10" \
            "amzn_decor 1 10" "amzn_diaper 5 10" "amzn_feeding 5 10" "amzn_furniture 2 4" \
            "amzn_gear 2 7" "amzn_gifts 2 4" "amzn_health 2 4" "amzn_media 2 10" \
            "amzn_moms 1 7" "amzn_pottytrain 1 1" "amzn_safety 2 7" \
            "amzn_strollers 2 10" "amzn_toys 5 7")

twenty_datasets=("nltcs 1 20" "kdd 2 20" "plants 5 20" "baudio 5 20" "jester 5 20" \
                "bnetflix 2 20" "accidents 1 20" "tretail 5 10" "pumsb_star 2 20" "dna 1 4" \
                "kosarek 5 10" "msweb 7 10" "book 5 10" "tmovie 5 20" "cwebkb 5 20" "cr52 5 20" \
                "c20ng 5 10" "bbc 5 20" "ad 7 1" "msnbc 1 10")
                

for params in "${amzn_datasets[@]}"
do
    set -- $params
    dataset=$1
    cluster_size=$2
    component_num=$3
    log_file_name="logs/SimplePGC_${cluster_size}_${component_num}_${dataset}.txt"
    output_model_path="models/SimplePGC_${dataset}.pt"
    if [ -e $log_file_name ]
    then
        echo "$log_file_name exists"
    else
        echo "Training on ${params}"
        python3.8 train.py --dataset_path /space/hzhang19/Density-Estimation-Datasets/datasets/ \
                --dataset $dataset --device cuda --cuda_core 0 --model SUM_DPP_MIX --max_epoch 50 \
                --batch_size 128 --lr 0.001 --weight_decay 0.001 \
                --max_cluster_size $cluster_size --component_num $component_num \
                --log_file $log_file_name --output_model_path $output_model_path
    fi
done