#!/bin/bash

amzn_datasets=("amzn_apparel" "amzn_bath" "amzn_bedding" "amzn_carseats" "amzn_decor" \
            "amzn_diaper" "amzn_feeding" "amzn_furniture" "amzn_gear" "amzn_gifts" \
            "amzn_health" "amzn_media" "amzn_moms" "amzn_pottytrain" "amzn_safety" \
            "amzn_strollers" "amzn_toys")

twenty_datasets=("nltcs" "kdd" "plants" "baudio" "jester" \
                "bnetflix" "accidents" "tretail" "pumsb_star" "dna" \
                "kosarek" "msweb" "book" "tmovie" "cwebkb" "cr52" \
                "c20ng" "bbc" "ad" "msnbc")


for dataset in "${amzn_datasets[@]}"
do
    model_path="models/SimplePGC_${dataset}.pt"
    if [ -e $model_path ]
    then
        python3.8 eval.py --dataset_path ../data/ \
                --dataset $dataset --device cuda --cuda_core 0 \
                --model_path $model_path --batch_size 128 
    else
        echo "Model for ${dataset} does not exist"
    fi

done