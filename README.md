# Distant Supervision for DIORA

This is the official repo for our EMNLP 2021 paper: 
Zhiyang Xu, [Andrew Drozdov<sup></sup>](https://mrdrozdov.github.io), [Jay Yoon Lee<sup></sup>](https://leejayyoon.github.io), [Tim O'Gorman<sup></sup>](https://timjogorman.github.io), [Subendhu Rongali<sup></sup>](https://subendhurongali.netlify.app), Dylan Finkbeiner, Shilpa Suresh, [Mohit Iyyer<sup></sup>](https://people.cs.umass.edu/~miyyer/) and [Andrew McCallum<sup></sup>](https://people.cs.umass.edu/~mccallum/), "Improved Latent Tree Induction with Distant Supervision via Span Constraints".

## Contents
1. [Setup](#Setup)
2. [Preparation](#Preparation)
3. [Training](#Training)
4. [Evaluation](#Evaluation)
5. [Related Works](#RelatedWorks)
6. [Citation](#Citation)


## Setup
1. Create environment
```
conda create -n s-diora python=3.6
```
2. Install Pytorch
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```
3. Install requirments
```
pip install -r requirements.txt
```
## Preparation
1. Prepare CoNLL2012 dataset

3. Prepare WSJ Penn Treebank dataset

4. Prepare MedMentions dataset

5. Prepare CRAFT dataset

## Training
1. Download pre-trained Diora model
```
mkdir ./download
cd ./download
wget http://diora-naacl-2019.s3.amazonaws.com/diora-checkpoints.zip
unzip diora-checkpoints.zip
```
2. A sample command line to train a model for PTB dataset
```
python main.py \
    --experiment_name pmi_seed3 \
    --default_experiment_directory ${EXP_DIR}/final_model/avg_pmi \
    --batch_size 1 \
    --accum_steps 1 \
    --validation_batch_size 128 \
    --lr 0.001 \
    --train_data_type wsj_emnlp \
    --train_filter_length 0 \
    --train_path ${DATA_DIR}/ptb/ptb-test-diora.parse \
    --validation_data_type wsj_emnlp \
    --validation_path /mnt/nfs/scratch1/zhiyangxu/co-diora-emnlp2021data/data/ptb-test.jsonl \
    --validation_filter_length 0 \
    --elmo_cache_dir ${DATA_DIR}/elmo \
    --emb elmo \
    --eval_after 1 \
    --eval_every_batch -1 \
    --eval_every_epoch 1 \
    --log_every_batch 100 \
    --max_step -1 \
    --max_epoch 40 \
    --opt adam \
    --save_after 0 \
    --num_warmup_steps 3000 \
    --load_model_path /mnt/nfs/scratch1/zhiyangxu/co-diora/experiment/real-world/log/avg_performance_v2_word2vec_390481271/model.best__parsing__f1.pt \
    --model_config '{"diora": {"normalize": "unit", "outside": true, "size": 400}}' \
    --eval_config '{"parsing": {"name": "eval-k1", "cky_mode": "cky", "enabled": true, "outside": false, "ground_truth": "/mnt/nfs/scratch1/zhiyangxu/co-diora-emnlp2021data/data/ptb-test.jsonl", "write":true, "scalars_key": "inside_s_components"}}' \
    --loss_config '{"reconstruct": {"path": "./resource/ptb_top_10k.txt", "weight": 1.0}}'
```
3. Training Args explanation

| Command | Values | Description |
| --- | --- | --- |
| `--experiment_name` | `str` | Name of the current experiment |
| `--default_experiment_directory` | `str` | Where to save the experiment |
| `--batch_size` | `int` | Size of the batch |
| `--accum_steps` | `int` | Accumulation steps before the optimizer takes a step |



## Evaludation

1. Download the best models reported in the paper

| Model Type | Performance | Constraints | Dataset |
| --- | --- | --- | --- |
| [NCBL<sup></sup>]() | 60.4 | `NER` | WSJ Penn Treebank |
| [MINDIFF<sup></sup>]() | 59.0 | `NER` | WSJ Penn Treebank |
| [RESCALE<sup></sup>]() | 61.9 | `NER` | WSJ Penn Treebank |
| [STRUCTURE RAMP<sup></sup>]() | 59.9 | `NER` | WSJ Penn Treebank |
| [NCBL<sup></sup>]() | 58.8 | `Gazatteer` | WSJ Penn Treebank |
| [NCBL<sup></sup>]() | 57.8 | `PMI` | WSJ Penn Treebank |
| [NCBL<sup></sup>]() | 56.8 | `NER` | CRAFT |

## RelatedWorks


## Citation

```
@inproceedings{diora2021emnlp,
  title={Improved Latent Tree Induction with Distant Supervision via Span Constraints},
  author={Zhiyang Xu, Andrew Drozdov, Jay Yoon Lee, Tim O'Gorman, Subendhu Rongali, Dylan Finkbeiner, Shilpa Suresh, Mohit Iyyer and Andrew McCallum},
  booktitle={EMNLP},
  year={2021},
}
```
