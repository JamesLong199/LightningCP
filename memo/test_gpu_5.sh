export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/memo_single_gpu --config configs/single_gpu.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/memo_single_gpu_no_estimate --config configs/single_gpu_no_estimate.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/memo_single_gpu_remove_ref_d01 --config configs/single_gpu_remove_ref_d01.yaml

# CUDA_VISIBLE_DEVICES=6 python inference_batch_hdtf.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/memo_deepcache_nonuniform --config configs/deepcache_nonuniform.yaml
# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py --audio_dir ../../data/HDTF/audios_30 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 --output_dir result_hdtf/memo_deepcache_nonuniform_sparse --config configs/deepcache_nonuniform_sparse.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/original_100 --config configs/default.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_deepcache_nonuniform_100 --config configs/deepcache_nonuniform.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_100 --config configs/single_gpu_no_estimate.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_bg_100 --config configs/single_gpu_no_estimate_bg.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_rr_u32_100 --config configs/single_gpu_no_estimate_rr_u32.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py --audio_dir ../../data/HDTF/audios_100 --image_dir ../../data/HDTF/images --mask_dir ../../data/HDTF/masks/resnet34 \
# --output_dir result_hdtf/memo_single_gpu_no_estimate_rr_u32_bg_100 --config configs/single_gpu_no_estimate_rr_u32_bg.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/memo_single_gpu_100 \
# --config configs/single_gpu_no_estimate.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/memo_single_gpu_bg_100 \
# --config configs/single_gpu_no_estimate_bg.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/memo_single_gpu_no_estimate_rr_u32_bg_100 \
# --config configs/single_gpu_no_estimate_rr_u32_bg.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/memo_deepcache_nonuniform_100 \
# --config configs/deepcache_nonuniform.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_fasterdiffusion_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/memo_fasterdiffusion_100 \
# --config configs/fasterdiffusion.yaml

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/memo_original_100 \
# --config configs/default.yaml


CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../data/MEAD/audios_W038_neu_1_003 \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/masks/resnet34 \
--output_dir result_mead/memo_single_gpu_W038 \
--config configs/single_gpu_no_estimate.yaml

CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../data/MEAD/audios_W038_neu_1_003 \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/masks/resnet34 \
--output_dir result_mead/memo_single_gpu_bg_W038 \
--config configs/single_gpu_no_estimate_bg.yaml

CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../data/MEAD/audios_W038_neu_1_003 \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/masks/resnet34 \
--output_dir result_mead/memo_single_gpu_no_estimate_rr_u32_bg_W038 \
--config configs/single_gpu_no_estimate_rr_u32_bg.yaml