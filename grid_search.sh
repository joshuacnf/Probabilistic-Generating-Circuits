#!/bin/bash

amzn_datasets=("amzn_apparel" "amzn_bath" "amzn_bedding" "amzn_carseats" "amzn_decor" \
            "amzn_diaper" "amzn_feeding" "amzn_furniture" "amzn_gear" "amzn_gifts" \
            "amzn_health" "amzn_media" "amzn_moms" "amzn_pottytrain" "amzn_safety" \
            "amzn_strollers" "amzn_toys")

st_datasets=("st_15_29_10016_0.01" "st_15_29_10016_0.02" "st_15_29_10016_0.03" \
            "st_15_29_10016_0.04" "st_15_29_10016_0.05" "st_15_29_10016_0.06" \
            "st_15_29_10016_0.07" "st_15_29_10016_0.08" "st_15_29_10016_0.09"
            "st_15_29_10016_0.10")

twenty_datasets=("nltcs" "kdd" "plants" "baudio" "jester" \
                "bnetflix" "accidents" "tretail" "pumsb_star" "dna" \
                "kosarek" "msweb" "book" "tmovie" "cwebkb" "cr52" \
                "c20ng" "bbc" "ad" "msnbc")

cluster_sizes=(1 2 3 5 8 10)
component_nums=(1 2 5 8 10 20)

for cluster_size in "${cluster_sizes[@]}"
do
    for component_num in "${component_nums[@]}"
    do
        for dataset in "${twenty_datasets[@]}"
        do
            echo "PGC Learning on ${dataset}"
            python3.8 train.py --dataset_path /space/hzhang19/Density-Estimation-Datasets/datasets/ \
                --dataset $dataset --device cuda --cuda_core 0 --model SUM_DPP_MIX --max_epoch 50 \
                --batch_size 256 --lr 0.001 --weight_decay 0.001 \
                --max_cluster_size $cluster_size --component_num $component_num \
                --log_file "logs/SimplePGC_${cluster_size}_${component_num}_${dataset}_0.001_log.txt" \
        done
    done
done