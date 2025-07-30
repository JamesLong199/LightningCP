export FFMPEG_PATH=ffmpeg-4.4-amd64-static

CUDA_VISIBLE_DEVICES=3 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_sfc_bg_rr_d01.yaml \
--audio_dir ../../data/MEAD/audios_100 \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/cropped_masks \
--output_dir result_mead/echomimic_sfc_bg_rr_d01_100 \
--batch_mode

CUDA_VISIBLE_DEVICES=3 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_acc_sfc_0135_bg_rr_d01.yaml \
--audio_dir ../../data/MEAD/audios_100 \
--image_dir ../../data/MEAD/images \
--mask_dir ../../data/MEAD/cropped_masks \
--output_dir result_mead/echomimic_acc_sfc_0135_bg_rr_d01_100 \
--batch_mode