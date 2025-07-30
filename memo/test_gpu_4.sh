#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120

# # Define the list of config names to process
# CONFIG_NAMES=(
#   "single_gpu_no_estimate_bg_spatial"
#   "single_gpu_no_estimate_bg_audio"
#   "single_gpu_no_estimate_rr_u31_bg"
#   "single_gpu_no_estimate_rr_u32_bg"
#   "single_gpu_no_estimate_rr_u3132"
# )

# # Fixed GPU ID
# GPU_ID=4

# # Set paths
# AUDIO_DIR="../../data/HDTF/audios_100"
# IMAGE_DIR="../../data/HDTF/images"
# MASK_DIR="../../data/HDTF/masks/resnet34"

# # Process each config sequentially
# for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
#   OUTPUT_DIR="result_hdtf/memo_${CONFIG_NAME}_100"
#   CONFIG_PATH="configs/${CONFIG_NAME}.yaml"

#   # Check if config file exists
#   if [ ! -f "$CONFIG_PATH" ]; then
#     echo "Error: Config file $CONFIG_PATH does not exist, skipping"
#     continue
#   fi

#   echo "----------------------------------------------"
#   echo "Running inference with:"
#   echo "  Config: $CONFIG_PATH"
#   echo "  Output directory: $OUTPUT_DIR"
#   echo "  GPU: $GPU_ID"
#   echo "----------------------------------------------"

#   # Run the inference command
#   CUDA_VISIBLE_DEVICES=$GPU_ID python inference_batch_hdtf_multi_gpu.py \
#     --audio_dir $AUDIO_DIR \
#     --image_dir $IMAGE_DIR \
#     --mask_dir $MASK_DIR \
#     --output_dir $OUTPUT_DIR \
#     --config $CONFIG_PATH
# done

# echo "All configs processed."

# CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_100 --config configs/single_gpu_no_estimate.yaml

# CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_bg_100 --config configs/single_gpu_no_estimate_bg.yaml

# CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_rr_u32_100 --config configs/single_gpu_no_estimate_rr_u32.yaml

# CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_rr_u32_bg_100 --config configs/single_gpu_no_estimate_rr_u32_bg.yaml

CUDA_VISIBLE_DEVICES=3 python inference_fasterdiffusion_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
--output_dir result_hdtf/memo_fasterdiffusion_100 --config configs/fasterdiffusion.yaml
