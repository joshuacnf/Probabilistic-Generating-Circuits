# Probabilistic Generating Circuits

This repo contains a PyTorch implementation for learning SimplePGC, which is 
the toy model used in the experiment section of the paper:

H. Zhang, B. Juba, G. Van den Broeck, 
**Probabilistic Generating Circuits**,
*ICML 2021*. 
https://arxiv.org/abs/2102.09768

SimplePGCs are one of the simplest classes of probabilistic generating 
circuits that are neither sum-product networks (SPNs) nor determinantal 
point processes (DPPs). This implementation for SimplePGC learns joint 
distributions over <em>binary</em> random variabels.


# Setup 

This will clone the repo, install a python virtual env (requires python 3.8), and
install the required packages.

    git clone https://github.com/joshuacnf/Probabilistic-Generating-Circuits.git
    cd Probabilistic-Generating-Circuits
    ./setup.sh

# Experiments

The `data` folder contains the datasets for the two density estimation benchmarks used in the paper: the Twenty Datasets and the Amazon Baby Registries. To reproduce the experiment results, run

    cd pgc
    mkdir logs models
    ./learn_best.sh

to learn SimplePGC for all datasets. Note that learning on all 
datasets may take a long time, and you can choose to learn SimplePGC for only a few 
datasets by modifying the script `learn_best.sh`.

Then we evaluate the stored SimplePGC models on the test sets via

    ./eval_best.sh,

which outputs the average test log-likelihoods for SimplePGC on each dataset.

# Try your own data
To learn a SimplePGC over a custom dataset called `persona`, we first add it to the `data` folder. 
Note that `persona` should be formated in the same way as other datasets from the benchmark: 
it should be a folder called `persona` containing three
files `persona.train.data`, `persona.valid.data` and `persona.test.data`, representing the train, valid
and test set, respectively; in the files, each line contains one binary sequence separated by comma.

Then we run the following (example) command to learn a SimplePGC and output the best model to
the path `./models/persona.pt`:

    python3.8 train.py --dataset_path ../data/
        --dataset persona --device cpu
        --model SimplePGC --max_epoch 50 \
        --batch_size 128 --lr 0.001 --weight_decay 0.001 \
        --max_cluster_size 8 --component_num 10 \
        --log_file SimplePGC_persona_8_10_log.txt --output_model_path ./models/persona.pt

If gpu is available, you might want to specify `--device cuda --cuda_core 0` to accelerate training.
Here, `--max_cluster_size` and `--component_num` are the two structural hyperparameters for SimplePGC, and
they can be tuned via grid search. Please refer to the paper and its appendix for further details.

Then we can evaluate the performance of the model on the test set by

    python3.8 eval.py --dataset_path ../data/ \
        --dataset persona --device cuda --cuda_core 0 \
        --model_path ./models/persona.pt --batch_size 128 
