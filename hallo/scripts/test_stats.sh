CUDA_VISIBLE_DEVICES=4 python scripts/inference_batch_hdtf.py \
--audio_dir ../../data/HDTF/audios_20 \
--image_dir ../../data/HDTF/images \
--mask_dir ../../data/HDTF/masks/resnet34 \
--output_dir test_stats/hallo_original \
--config configs/inference/original.yaml 