#!/bin/bash
#SBATCH -J eval_mdlm_sdc_owt_sample_steps
#SBATCH -o sample_folder/%x_%j.out
#SBATCH -e sample_error/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:gpu:1
#SBATCH --time=06:00:00
#SBATCH -p defq

ROOT_DIR="/home/at0842/ycl466704.ai13/gqop0919"
export HF_HOME=${ROOT_DIR}/hf_cache
export HF_DATASETS_CACHE=${ROOT_DIR}/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=${ROOT_DIR}/hf_cache/hub
export TRANSFORMERS_CACHE=${ROOT_DIR}/hf_cache/transformers
export HF_DATASETS_TRUST_REMOTE_CODE=1

for steps in 64 128 256 512 768 1000  
do
  echo "========================================"
  echo "Running sample_eval with sampling.steps=${steps}"
  echo "========================================"

  python main.py \
    mode=sample_eval \
    data.cache_dir=${ROOT_DIR}/hf_cache/datasets \
    hydra.run.dir=${ROOT_DIR}/mdlm-sdc/sample_output/sample_steps_${steps}_run-$(date +%Y%m%d-%H%M%S) \
    eval.checkpoint_path=${ROOT_DIR}/mdlm-sdc/outputs/run-20260620-202646_conf0.3_sratio0.5_20000/checkpoints/last.ckpt \
    data=openwebtext-split \
    model.length=1024 \
    sampling.predictor=ddpm_cache \
    sampling.steps=${steps} \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=200 \
    backbone=dit \
    +wandb.offline=true
done