export FFMPEG_PATH=ffmpeg-4.4-amd64-static

# faster diffusion ...
CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_fasterdiffusion.py \
--config configs/prompts/fasterdiffusion.yaml \
--audio_dir ../../data/MEAD/audios_100 \
--image_dir ../../data/MEAD/images \
--output_dir result_mead/echomimic_fasterdiffusion_100 \
--batch_mode

CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_fasterdiffusion.py \
--config configs/prompts/fasterdiffusion_acc.yaml \
--audio_dir ../../data/MEAD/audios_100 \
--image_dir ../../data/MEAD/images \
--output_dir result_mead/echomimic_fasterdiffusion_acc_100 \
--batch_mode




