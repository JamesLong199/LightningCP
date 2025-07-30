export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/memo_original \
# --config configs/default.yaml \
# --log_latency True

# CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/memo_deepcache_nonuniform \
# --config configs/deepcache_nonuniform.yaml \
# --log_latency True

# CUDA_VISIBLE_DEVICES=4,5,6 python inference_fasterdiffusion_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/memo_fasterdiffusion \
# --config configs/fasterdiffusion.yaml \
# --log_latency True

CUDA_VISIBLE_DEVICES=4,5,6 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../data/MEAD/audios_test \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/masks/resnet34 \
--output_dir test_latency/memo_multi_gpu \
--config configs/single_gpu_no_estimate.yaml \
--log_latency True

CUDA_VISIBLE_DEVICES=4,5,6 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../data/MEAD/audios_test \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/masks/resnet34 \
--output_dir test_latency/memo_multi_gpu_bg \
--config configs/single_gpu_no_estimate_bg.yaml \
--log_latency True


CUDA_VISIBLE_DEVICES=4,5,6 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../data/MEAD/audios_test \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/masks/resnet34 \
--output_dir test_latency/memo_multi_gpu_rr_u32_bg \
--config configs/single_gpu_no_estimate_rr_u32_bg.yaml \
--log_latency True
