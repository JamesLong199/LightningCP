CUDA_VISIBLE_DEVICES=5 python inference_batch_hdtf.py \
--audio_dir ../../data/HDTF/audios_20 \
--image_dir ../../data/HDTF/images \
--mask_dir ../../data/HDTF/masks/resnet34 \
--output_dir test_stats/memo_original \
--config configs/default.yaml 