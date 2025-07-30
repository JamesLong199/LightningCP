export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120

CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf.py \
--audio_dir ../../../data/MEAD/audios_test \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/memo_original \
--config configs/default.yaml \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf.py \
--audio_dir ../../../data/MEAD/audios_test \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/memo_deepcache_nonuniform \
--config configs/deepcache_nonuniform.yaml \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python inference_fasterdiffusion_batch_hdtf_multi_gpu.py \
--audio_dir ../../../data/MEAD/audios_test \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/memo_fasterdiffusion \
--config configs/fasterdiffusion.yaml \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../../data/MEAD/audios_test \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/memo_multi_gpu \
--config configs/single_gpu_no_estimate.yaml \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../../data/MEAD/audios_test \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/memo_multi_gpu_bg \
--config configs/single_gpu_no_estimate_bg.yaml \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python inference_batch_hdtf_multi_gpu.py \
--audio_dir ../../../data/MEAD/audios_test \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/memo_multi_gpu_rr_u32_bg \
--config configs/single_gpu_no_estimate_rr_u32_bg.yaml \
--log_flops True
