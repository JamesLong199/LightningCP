# CUDA_VISIBLE_DEVICES=4,5 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform --config configs/inference/multi_gpu_no_estimate.yaml
# CUDA_VISIBLE_DEVICES=4,5 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform --config configs/inference/multi_gpu.yaml
# CUDA_VISIBLE_DEVICES=4,5 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_bg --config configs/inference/multi_gpu_bg.yaml
# CUDA_VISIBLE_DEVICES=4,5 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_all --config configs/inference/multi_gpu_all.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu --config configs/inference/single_gpu.yaml
# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_masked_attention --config configs/inference/single_gpu_masked_attention.yaml
# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_masked_attention_remove_ref --config configs/inference/single_gpu_masked_attention_remove_ref.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_remove_ref_u32_bg --config configs/inference/single_gpu_remove_ref_u32_bg.yaml
# CUDA_VISIBLE_DEVICES=5 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_remove_ref_d01_bg --config configs/inference/single_gpu_remove_ref_d01_bg.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_bg --config configs/inference/single_gpu_bg.yaml


# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_bg_spatial --config configs/inference/single_gpu_bg_spatial.yaml
# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_bg_audio --config configs/inference/single_gpu_bg_audio.yaml
# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_bg_temporal --config configs/inference/single_gpu_bg_temporal.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_faster_diffusion.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_fasterdiffusion_100 --config configs/inference/fasterdiffusion.yaml
# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_faster_diffusion.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/hallo_fasterdiffusion_pni --config configs/inference/fasterdiffusion_pni.yaml

# CUDA_VISIBLE_DEVICES=3 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/hallo_sfc_nonuniform_single_gpu_bg_100 \
# --config configs/inference/single_gpu_bg.yaml

CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_faster_diffusion.py \
--audio_dir ../../../data/MEAD/audios_100 \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir result_mead/hallo_fasterdiffusion_100 \
--config configs/inference/fasterdiffusion.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/hallo_sfc_nonuniform_single_gpu_100 \
# --config configs/inference/single_gpu.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/hallo_sfc_nonuniform_single_gpu_bg_100 \
# --config configs/inference/single_gpu_bg.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/hallo_sfc_nonuniform_single_gpu_remove_ref_d01_bg_100 \
# --config configs/inference/single_gpu_remove_ref_d01_bg.yaml

CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf.py \
--audio_dir ../../../data/MEAD/audios_100 \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir result_mead/hallo_deepcache_nonuniform_100 \
--config configs/inference/deepcache_nonuniform.yaml

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_faster_diffusion.py \
# --audio_dir ../../data/MEAD/audios_W038_neu_1_003 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/hallo_fasterdiffusion_W038 \
# --config configs/inference/fasterdiffusion.yaml

CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../data/MEAD/audios_100 \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/masks/resnet34 \
--output_dir result_mead/hallo_sfc_nonuniform_single_gpu_remove_ref_d01_100 \
--config configs/inference/single_gpu_remove_ref_d01.yaml