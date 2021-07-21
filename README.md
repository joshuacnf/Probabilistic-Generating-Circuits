# Probabilistic Generating Circuits

PyTorch implementation for learning SimplePGC, which is the toy model used in the experiment section of the paper:

H. Zhang, B. Juba, G. Van den Broeck,
**Probabilistic Generating Circuits**,
*ICML 2021*.

SimplePGCs are one of the simplest
class of probabilistic generating circuits that are neither sum-product networks (SPNs) nor determinantal point processes (DPPs).


# Setup 

This will clone the repo, install a python virtual env (requires python 3.8), and
install the required packages.

    git clone https://github.com/joshuacnf/Probabilistic-Generating-Circuits.git
    cd Probabilistic-Generating-Circuits
    ./setup.sh

# Experiments

To reproduce the experiment results in the paper:

We first learn the SimplePGC models:
    cd pgc
    mkdir logs models
    ./learn_best.sh

Then we evaluate the stored SimplePGC models via:
    ./eval_best.sh

# Try your own data
Suppose we're given custom dataset 

    python3.8 train.py --dataset_path ../data/
        --dataset $dataset --device cuda --cuda_core 0 
        --model SimplePGC --max_epoch 50 \
        --batch_size 128 --lr 0.001 --weight_decay 0.001 \
        --max_cluster_size $cluster_size --component_num $component_num \
        --log_file $log_file_name --output_model_path ./models/${dataset}.pt

