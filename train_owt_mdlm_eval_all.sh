#!/bin/bash
#SBATCH -J eval_mdlm_sdc_all_seed1to5
#SBATCH -o eval_folder/%x_%j.out
#SBATCH -e eval_error/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:gpu:1
#SBATCH --time=06:00:00
#SBATCH -p defq

set -euo pipefail

ROOT_DIR="/home/at0842/ycl466704.ai13/gqop0919"
export HF_HOME=${ROOT_DIR}/hf_cache
export HF_DATASETS_CACHE=${ROOT_DIR}/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=${ROOT_DIR}/hf_cache/hub
export TRANSFORMERS_CACHE=${ROOT_DIR}/hf_cache/transformers
export HF_DATASETS_TRUST_REMOTE_CODE=1

# 讓錯誤訊息更完整
export HYDRA_FULL_ERROR=1

BASE_OUTDIR=${ROOT_DIR}/mdlm-sdc/eval_output
STAMP=$(date +%Y%m%d-%H%M%S)

DATASETS=(
  openwebtext-split
  scientific_papers_arxiv
  wikitext103
  lambada
  ptb
  scientific_papers_pubmed
)

SEEDS=(1 2 3 4 5)

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo "[SLURM_JOB_ID=${SLURM_JOB_ID}] Running ppl_eval"
    echo "DATASET=${DATASET}"
    echo "SEED=${SEED}"
    echo "========================================"

    python main.py \
      mode=ppl_eval \
      seed=${SEED} \
      loader.batch_size=16 \
      loader.eval_batch_size=16 \
      model=small \
      data=${DATASET} \
      parameterization=subs \
      backbone=dit \
      model.length=1024 \
      trainer.max_steps=10000 \
      data.cache_dir=${ROOT_DIR}/hf_cache/datasets \
      hydra.run.dir=${BASE_OUTDIR}/ppl_eval_${DATASET}_${STAMP}_job${SLURM_JOB_ID}_seed${SEED} \
      eval.checkpoint_path=${ROOT_DIR}/mdlm-sdc/outputs/run-20260620-202646_conf0.3_sratio0.5_20000/checkpoints/last.ckpt \
      eval.compute_generative_perplexity=false \
      +wandb.offline=true
  done
done