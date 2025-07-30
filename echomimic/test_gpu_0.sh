export FFMPEG_PATH=ffmpeg-4.4-amd64-static

CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch.py \
--config configs/prompts/animation_batch_acc_dc_0135.yaml \
--audio_dir ../../data/MEAD/audios_100 \
--image_dir ../../data/MEAD/images \
--output_dir result_mead/echomimic_acc_dc_0135_100 \
--batch_mode