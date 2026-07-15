#!/bin/bash
#SBATCH -J train_mdlm_openwebtext-split
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e error/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:gpu:1
#SBATCH --time=48:00:00
#SBATCH -p h200q

ROOT_DIR="/home/at0842/ycl466704.ai13/gqop0919"
export HF_HOME=${ROOT_DIR}/hf_cache
export HF_DATASETS_CACHE=${ROOT_DIR}/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=${ROOT_DIR}/hf_cache/hub
export TRANSFORMERS_CACHE=${ROOT_DIR}/hf_cache/transformers
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ===== Training hyperparameters =====
BATCH_SIZE=16
EVAL_BATCH_SIZE=16
MAX_STEPS=20000
MODEL_LENGTH=1024

LR_WARMUP_STEPS=2500
SDC_KL_WARMUP_STEPS=5000
SDC_S_RATIO=0.5
SDC_CONFIDENCE_THRESHOLD=0.0
SDC_CE_COEF=1.0
SDC_KL_LAMBDA=1.0

FROM_STEP=0
TO_STEP=${MAX_STEPS}

# ===== Name formatting =====
fmt() {
  printf "%g" "$1"
}

CE_NAME=$(fmt "$SDC_CE_COEF")
KL_NAME=$(fmt "$SDC_KL_LAMBDA")
SRATIO_NAME=$(fmt "$SDC_S_RATIO")
CONF_NAME=$(fmt "$SDC_CONFIDENCE_THRESHOLD")
DATE_TAG=$(date +%m%d%H%M)

WANDB_NAME="mdlm-owt-CE${CE_NAME}-KL${KL_NAME}-sratio${SRATIO_NAME}-conf${CONF_NAME}-kl${SDC_KL_LAMBDA}-from${FROM_STEP}to${TO_STEP}-m${DATE_TAG}"

RUN_DIR="${ROOT_DIR}/mdlm-sdc/outputs/run-$(date +%Y%m%d-%H%M%S)_conf${CONF_NAME}_sratio${SRATIO_NAME}_KL${SDC_KL_LAMBDA}"

python main.py \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${EVAL_BATCH_SIZE} \
  model=small \
  data=openwebtext-split \
  wandb.name=${WANDB_NAME} \
  parameterization=subs \
  model.length=${MODEL_LENGTH} \
  trainer.max_steps=${MAX_STEPS} \
  data.cache_dir=${ROOT_DIR}/hf_cache/datasets \
  hydra.run.dir=${RUN_DIR} \
  trainer.val_check_interval=50000 \
  trainer.gradient_clip_val=1.0 \
  optim.lr=3e-4 \
  lr_scheduler.num_warmup_steps=${LR_WARMUP_STEPS} \
  training.sdc_kl_warmup_steps=${SDC_KL_WARMUP_STEPS} \
  training.sdc_s_ratio=${SDC_S_RATIO} \
  training.sdc_confidence_threshold=${SDC_CONFIDENCE_THRESHOLD} \
  training.sdc_ce_coef=${SDC_CE_COEF} \
  training.sdc_kl_lambda=${SDC_KL_LAMBDA}
  