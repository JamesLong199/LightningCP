# LightningCP

## Installation

Follow the instructions of [Hallo](https://github.com/JamesLong199/LightningCP/tree/main/hallo#%EF%B8%8F-installation), [MEMO](https://github.com/JamesLong199/LightningCP/tree/main/memo#installation), and [EchoMimic](https://github.com/JamesLong199/LightningCP/tree/main/echomimic#%EF%B8%8F-installation) to install environments and download pretrained models.

## Run Demos

**Tested on NVIDIA A800 GPUs with 80GB VRAM.**

---

## Hallo

### Setup

```bash
cd Hallo
conda activate hallo
```

### Base Model Inference

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_batch_hdtf.py \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/masks \
    --output_dir ../results/hallo_original \
    --config configs/inference/original.yaml
```

### LCP Single GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_batch_hdtf_multi_gpu.py \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/masks \
    --output_dir ../results/hallo_single_gpu \
    --config configs/inference/single_gpu_remove_ref_d01_bg.yaml
```

### LCP Multi-GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/inference_batch_hdtf_multi_gpu.py \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/masks \
    --output_dir ../results/hallo_multi_gpu \
    --config configs/inference/multi_gpu_remove_ref_d01_bg.yaml
```

---

## MEMO

### Setup

```bash
cd memo
conda activate memo
```

### Base Model Inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference_batch_hdtf.py \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/masks \
    --output_dir ../results/memo_original \
    --config configs/default.yaml
```

### LCP Single GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference_batch_hdtf_multi_gpu.py \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/masks \
    --output_dir ../results/memo_single_gpu \
    --config configs/single_gpu_no_estimate_rr_u32_bg.yaml
```

### LCP Multi-GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python inference_batch_hdtf_multi_gpu.py \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/masks \
    --output_dir ../results/memo_multi_gpu \
    --config configs/multi_gpu_no_estimate_rr_u32_bg.yaml
```

---

## EchoMimic

### Setup

```bash
cd echomimic
conda activate echomimic
export FFMPEG_PATH=ffmpeg-4.4-amd64-static
```

### Base Model Inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch.py \
    --config configs/prompts/animation_batch.yaml \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --output_dir ../results/echomimic_original \
    --batch_mode
```

### LCP Single GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
    --config configs/prompts/animation_batch_sfc_bg_rr_d01.yaml \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/cropped_masks \
    --output_dir test_flops/echomimic_single_gpu \
    --batch_mode
```

### LCP Multi-GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python infer_audio2vid_batch_multi_gpu.py \
    --config configs/prompts/animation_batch_sfc_bg_rr_d01_multi_gpu.yaml \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/cropped_masks \
    --output_dir test_flops/echomimic_multi_gpu \
    --batch_mode
```

---

## EchoMimic Acc

### Setup

```bash
cd echomimic
conda activate echomimic
export FFMPEG_PATH=ffmpeg-4.4-amd64-static
```

### Base Model Inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch.py \
    --config configs/prompts/animation_batch_acc.yaml \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --output_dir ../results/echomimic_acc \
    --batch_mode
```

### LCP Single GPU Inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
    --config configs/prompts/animation_batch_sfc.yaml \
    --audio_dir ../examples/audios \
    --image_dir ../examples/images \
    --mask_dir ../examples/cropped_masks \
    --output_dir test_flops/echomimic_acc_single_gpu \
    --batch_mode
```

---
