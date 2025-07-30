# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/hallo_original \
# --config configs/inference/original.yaml \
# --log_latency True \

# CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/hallo_deepcache_nonuniform \
# --config configs/inference/deepcache_nonuniform.yaml \
# --log_latency True

# CUDA_VISIBLE_DEVICES=4,5,6 python scripts/inference_batch_fasterdiffusion_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --output_dir test_latency/hallo_fasterdiffusion_multi_gpu \
# --config configs/inference/fasterdiffusion_multi_gpu.yaml \
# --log_latency

# CUDA_VISIBLE_DEVICES=4,5,6 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/hallo_sfc_nonuniform_multi_gpu \
# --config configs/inference/single_gpu.yaml \
# --log_latency True

# CUDA_VISIBLE_DEVICES=4,5,6 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/hallo_sfc_nonuniform_multi_gpu_bg \
# --config configs/inference/single_gpu_bg.yaml \
# --log_latency True

# CUDA_VISIBLE_DEVICES=4,5,6 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/hallo_sfc_nonuniform_multi_gpu_rr_d01_bg \
# --config configs/inference/single_gpu_remove_ref_d01_bg.yaml \
# --log_latency True

# CUDA_VISIBLE_DEVICES=4,5,6 python scripts/inference_batch_hdtf_multi_gpu.py \
# --audio_dir ../../data/MEAD/audios_test \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir test_latency/hallo_sfc_nonuniform_multi_gpu_rr_d01 \
# --config configs/inference/single_gpu_remove_ref_d01.yaml \
# --log_latency True



