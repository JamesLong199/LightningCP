
export FFMPEG_PATH=ffmpeg-4.4-amd64-static

CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch.py \
--config configs/prompts/animation_batch.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--output_dir test_flops/echomimic_original \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=6 python infer_audio2vid_batch.py \
--config configs/prompts/animation_batch_dc_nonuniform.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--output_dir test_flops/echomimic_dc_nonuniform \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch.py \
--config configs/prompts/animation_batch_acc.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--output_dir test_flops/echomimic_acc \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch.py \
--config configs/prompts/animation_batch_acc_dc_0135.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--output_dir test_flops/echomimic_acc_dc_0135 \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_fasterdiffusion.py \
--config configs/prompts/fasterdiffusion.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--output_dir test_flops/echomimic_fasterdiffusion \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_fasterdiffusion.py \
--config configs/prompts/fasterdiffusion_acc.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--output_dir test_flops/echomimic_fasterdiffusion_acc \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_sfc.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/echomimic_sfc \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_acc_sfc_0135.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/echomimic_acc_sfc_0135 \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_sfc_bg.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/echomimic_sfc_bg \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_acc_sfc_0135_bg.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/echomimic_acc_sfc_0135_bg \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=0 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_sfc_bg_rr_d01.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/echomimic_sfc_bg_rr_d01 \
--batch_mode \
--log_flops True

CUDA_VISIBLE_DEVICES=4 python infer_audio2vid_batch_multi_gpu.py \
--config configs/prompts/animation_batch_acc_sfc_0135_bg_rr_d01.yaml \
--audio_dir ../../../data/MEAD/audios_test_echomimic \
--image_dir ../../../data/MEAD/images \
--mask_dir ../../../data/MEAD/masks/resnet34 \
--output_dir test_flops/echomimic_acc_sfc_0135_bg_rr_d01 \
--batch_mode \
--log_flops True