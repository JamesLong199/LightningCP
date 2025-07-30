export FFMPEG_PATH=ffmpeg-4.4-amd64-static

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_acc.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_original \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_dc_int2.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_dc_int2 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_dc_int3.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_dc_int3 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_dc_nonuniform.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_dc_nonuniform \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_acc_dc_012.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_dc_012 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_acc_dc_014.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_dc_014 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_acc_dc_015.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_dc_015 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_012.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_012 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_014.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_014 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_015.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_015 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_no_estimate.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_no_estimate \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_bg.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_bg \
# --batch_mode

# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_rr_d00.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_rr_d00 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_rr_d01.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_rr_d01 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_rr_u32.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_rr_u32 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_rr_u30.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_rr_u30 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_rr_u31.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_rr_u31 \
# --batch_mode



# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0123.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0123 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0124.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0124 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0125.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0125 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0134.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0134 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0135.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0135 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0145.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0145 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_014_no_estimate.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_014_no_estimate \
# --batch_mode


# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_rr_d01.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_rr_d01_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_dc_nonuniform.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_dc_nonuniform_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_bg_spatial.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_bg_spatial \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_bg_temporal.yaml \
# --audio_dir ../../data/HDTF/audios_30 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_bg_temporal \
# --batch_mode

# CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0135.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0135_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0135.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/echomimic_acc_sfc_0135_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_acc_dc_0135.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_dc_0135_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_dc_nonuniform.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --output_dir result_mead/echomimic_dc_nonuniform_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/masks/resnet34 \
# --output_dir result_mead/echomimic_sfc_100 \
# --batch_mode

CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch_fasterdiffusion.py \
--config configs/prompts/fasterdiffusion.yaml \
--audio_dir ../../data/HDTF/audios_100 \
--image_dir ../../data/HDTF/images \
--output_dir result_hdtf/echomimic_fasterdiffusion_100 \
--batch_mode

# CUDA_VISIBLE_DEVICES=3 python infer_audio2vid_batch_fasterdiffusion.py \
# --config configs/prompts/fasterdiffusion_acc.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_fasterdiffusion_acc_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=3 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --output_dir result_mead/echomimic_original_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=3 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_acc.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --output_dir result_mead/echomimic_acc_100 \
# --batch_mode

# faster diffusion ...
CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch_fasterdiffusion.py \
--config configs/prompts/fasterdiffusion.yaml \
--audio_dir ../../data/MEAD/audios_100 \
--image_dir ../../data/MEAD/images \
--output_dir result_mead/echomimic_fasterdiffusion_100 \
--batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_fasterdiffusion.py \
# --config configs/prompts/fasterdiffusion_acc.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --output_dir result_mead/echomimic_fasterdiffusion_acc_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_bg.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --mask_dir ../../data/HDTF/cropped_masks \
# --output_dir result_hdtf/echomimic_sfc_bg_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0135_bg.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --mask_dir ../../data/HDTF/cropped_masks \
# --output_dir result_hdtf/echomimic_acc_sfc_0135_bg_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_bg_rr_d01.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_sfc_bg_rr_d01_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0135_bg_rr_d01.yaml \
# --audio_dir ../../data/HDTF/audios_100 \
# --image_dir ../../data/HDTF/images \
# --output_dir result_hdtf/echomimic_acc_sfc_0135_bg_rr_d01_100 \
# --batch_mode


# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_bg.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/cropped_masks \
# --output_dir result_mead/echomimic_sfc_bg_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=5 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0135_bg.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --mask_dir ../../data/MEAD/cropped_masks \
# --output_dir result_mead/echomimic_acc_sfc_0135_bg_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_sfc_bg_rr_d01.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --output_dir result_mead/echomimic_sfc_bg_rr_d01_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
# --config configs/prompts/animation_batch_acc_sfc_0135_bg_rr_d01.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --output_dir result_mead/echomimic_acc_sfc_0135_bg_rr_d01_100 \
# --batch_mode

# CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch.py \
# --config configs/prompts/animation_batch_acc_dc_0135.yaml \
# --audio_dir ../../data/MEAD/audios_100 \
# --image_dir ../../data/MEAD/images \
# --output_dir result_mead/echomimic_acc_dc_0135_100 \
# --batch_mode