# MDLM-SDC 使用說明

本文件說明如何建立執行環境、下載 checkpoint、進行訓練，以及執行 Perplexity 與 Generative Perplexity 評估。

---

# 1. 環境安裝

## 1.1 進入 `mdlm-sdc` 資料夾

```bash
cd mdlm-sdc
```

## 1.2 建立 Conda 環境

```bash
conda env create -f mdlm_environment.yaml
```

## 1.3 啟用環境

```bash
conda activate mdlm
```

## 1.4 安裝額外套件

以下 wheel 對應：

- Python 3.9
- PyTorch 2.2
- CUDA 11.8
- CXX11 ABI = False
- Linux x86_64

建議使用 `python -m pip`，確保套件安裝到目前啟用的 Conda 環境。

[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases?page=3)

下載causal_conv1d-1.1.3.post1+cu118torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

[mamba](https://github.com/state-spaces/mamba/releases?page=3)

下載mamba_ssm-1.1.4+cu118torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

[flash-attention](https://github.com/Dao-AILab/flash-attention/releases?page=6)

下載flash_attn-2.5.6+cu118torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

```bash
python -m pip install   causal_conv1d-1.1.3.post1+cu118torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

python -m pip install   mamba_ssm-1.1.4+cu118torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
 
python -m pip install   flash_attn-2.5.6+cu118torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```
---

# 2. Checkpoint

Checkpoint 下載位置：

[gqop/mdlm-sdc-ckpt](https://huggingface.co/gqop/mdlm-sdc-ckpt/tree/main)

## 2.1 資料夾名稱說明

例如：

```text
run-20260617-211010_conf0_sratio0.5_KL0_20000
```

各欄位意義如下：

- `run-20260617-211010`：執行日期與時間。
- `conf0`：confidence threshold 為 `0`。
- `sratio0.5`：低遮罩層級 `s` 為高遮罩層級 `t` 的 `0.5` 倍。
- `KL0`：KL consistency loss 的係數為 `0`。
- `20000`：訓練步數或該實驗設定所標示的 step 數。

Checkpoint 資料夾由上到下依序包含：

1. 原始 MDLM，不使用 SDC。
2. MDLM-SDC，不使用 confidence threshold 篩選。
3. MDLM-SDC，使用不同 confidence threshold 的實驗結果。

將checkpoint放入outputs資料夾中

---

# 3. 執行前準備

先進入專案資料夾：

```bash
cd mdlm-sdc
```

建立訓練、評估與輸出資料夾：

```bash
mkdir -p   error   watch_folder   eval_error   eval_folder   eval_output   sample_error   sample_folder   sample_output   outputs
```
---

# 4. 模型訓練

使用：

```text
train_owt_mdlm.sh
```

## 4.1 設定 `ROOT_DIR`

先修改腳本中的：

```bash
ROOT_DIR="/你的絕對路徑"
```

`ROOT_DIR` 應指向包含 `mdlm-sdc` 資料夾的上一層目錄。

例如專案完整路徑為：

```text
/home/at0842/ycl466704.ai13/gqop0919/mdlm-sdc
```

則設定為：

```bash
ROOT_DIR="/home/at0842/ycl466704.ai13/gqop0919"
```

## 4.2 提交訓練工作

確認目前位於 `mdlm-sdc` 資料夾後執行：

```bash
sbatch train_owt_mdlm.sh
```

查看工作狀態：

```bash
squeue
```

## 4.3 訓練參數說明

- `SDC_KL_WARMUP_STEPS=5000`  
  KL consistency loss 的 warmup 步數。前 5000 steps 逐漸提高 KL loss 權重，避免訓練初期受到過強約束。

- `SDC_S_RATIO=0.5`  
  控制較低遮罩層級 `s` 與較高遮罩層級 `t` 的比例，通常表示 `s = 0.5 × t`。

- `SDC_CONFIDENCE_THRESHOLD=0.0`  
  Teacher 預測的信心門檻。設為 `0.0` 表示不篩選，所有對齊位置都會參與 consistency loss。

- `SDC_CE_COEF=1.0`  
  原始 MDLM cross-entropy reconstruction loss 的權重。

- `SDC_KL_LAMBDA=1.0`  
  SDC KL consistency loss 的最大權重；warmup 結束後使用此係數。

---

# 5. Perplexity 評估

使用：

```text
train_owt_mdlm_eval_all.sh
```

## 5.1 修改路徑

先修改腳本中的：

```bash
ROOT_DIR="/你的絕對路徑"
```

並依照 checkpoint 實際位置修改：

```bash
eval.checkpoint_path=/你的/checkpoint/路徑/last.ckpt
```

## 5.2 提交評估工作

```bash
sbatch train_owt_mdlm_eval_all.sh
```

工作完成後，log 會輸出至：

```text
eval_folder/
```

## 5.3 統計不同 Seed 的 Perplexity

使用：

```text
summarize_ppl.py
```

執行方式：

```bash
python summarize_ppl.py   eval_folder/eval_mdlm_sdc_all_seed1to5_59410.out
```

請將檔名替換成實際產生的 `.out` 檔案。

程式會依照資料集整理不同 seed 的 `val/ppl`，並計算：

- 各 seed 的 Perplexity
- 平均值
- 標準差

---

# 6. Generative Perplexity 評估

使用：

```text
train_owt_mdlm_eval_genppl.sh
```

## 6.1 修改路徑

先修改腳本中的：

```bash
ROOT_DIR="/你的絕對路徑"
```

並依照 checkpoint 實際位置修改：

```bash
eval.checkpoint_path=/你的/checkpoint/路徑/last.ckpt
```

## 6.2 提交評估工作

```bash
sbatch train_owt_mdlm_eval_genppl.sh
```

工作完成後，log 會輸出至：

```text
sample_folder/
```

## 6.3 整理生成評估結果

使用：

```text
summarize_genppl.py
```

執行方式：

```bash
python summarize_genppl.py   sample_folder/eval_mdlm_sdc_owt_sample_steps_59414.out
```

請將檔名替換成實際產生的 `.out` 檔案。

程式會依照不同的 `sampling.steps` 列出：

- Generative Perplexity
- Generation Entropy
- Distinct-4

---

# 7. 常用指令

查看目前的 Slurm 工作：

```bash
squeue -u "$USER"
```

取消工作：

```bash
scancel <JOB_ID>
```

查看輸出檔案：

```bash
tail -f eval_folder/檔案名稱.out
```

查看錯誤訊息：

```bash
tail -f eval_error/檔案名稱.err
```

確認目前 Conda 環境：

```bash
echo "$CONDA_DEFAULT_ENV"
which python
```
