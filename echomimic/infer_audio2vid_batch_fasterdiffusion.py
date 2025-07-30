#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EchoMimic
@File    ：audio2vid.py
@Author  ：juzhen.czy
@Date    ：2024/3/4 17:43 
'''
import argparse
import os

import random
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
# from src.models.unet_3d_echo_parallel import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
# from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
# from src.pipelines.pipeline_echo_mimic_reordered import Audio2VideoPipeline
# from src.pipelines.pipeline_echo_mimic_multi_gpu import Audio2VideoPipeline
from src.pipelines.pipeline_fasterdiffusion import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN

from src.models.masking_utils import load_mask_fg
import copy
import time
from tqdm import tqdm
import glob

from contextlib import nullcontext
from torch.utils.flop_counter import FlopCounterMode

from src.models.utils_fasterdiffusion import register_faster_forward_multi_gpu as register_faster_forward

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/animation.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--facemusk_dilation_ratio", type=float, default=0.1)
    parser.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)

    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")

    # Add batch processing arguments
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--image_dir", type=str, help="Directory containing image files")
    parser.add_argument("--output_dir", type=str, help="Directory to save output videos")
    parser.add_argument("--batch_mode", action="store_true", help="Process files in batch mode")
    parser.add_argument("--log_latency", default=False, help="Log latency")
    parser.add_argument("--log_flops", default=False, help="Log flops")
    
    args = parser.parse_args()

    return args

def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]


# Modified function to initialize only denoising_unet on secondary GPUs
def initialize_models_on_gpus(denoising_unet, gpu_ids, weight_dtype):
    """
    Initialize denoising_unet on GPUs other than GPU 0.
    All models remain on GPU 0, only denoising_unet is copied to other GPUs.
    """
    print(f"Initializing denoising_unet on GPUs: {gpu_ids[1:] if len(gpu_ids) > 1 else 'None'}")
    
    # Dictionary to store denoising_unet models for each GPU
    gpu_models = {}
    
    # Skip GPU 0 as all models are already there
    for gpu_id in gpu_ids[1:]:
        device = f"cuda:{gpu_id}"
        print(f"Initializing denoising_unet on {device}")
        
        # Create a deep copy of denoising_unet for this GPU
        gpu_denoising_unet = copy.deepcopy(denoising_unet)
        
        # Move to the appropriate GPU
        # gpu_denoising_unet.to(device, dtype=weight_dtype, non_blocking=True)
        gpu_denoising_unet.to(device, dtype=weight_dtype)
                
        # Store in dictionary
        gpu_models[gpu_id] = gpu_denoising_unet
    
    return gpu_models

def get_file_basename(file_path):
    """Extract the base filename without extension from a file path."""
    return os.path.splitext(os.path.basename(file_path))[0]

def process_single_pair(
    ref_image_path,
    audio_path,
    output_dir,
    pipe,
    face_detector,
    args,
    config,
    generator,
    weight_dtype,
    log_file,
    log_latency,
    log_flops,
):
    ref_name = Path(ref_image_path).stem
    audio_name = Path(audio_path).stem
    time_str = datetime.now().strftime("%H%M%S")
    final_fps = args.fps
    width, height = args.W, args.H

    # Create base output filename
    base_output_name = f"{audio_name}"
    
    # Process face mask
    face_img = cv2.imread(ref_image_path)
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)

    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4]
        xyxy = np.round(xyxy).astype('int')
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * args.facemusk_dilation_ratio)
        c_pad = int((ce - cb) * args.facemusk_dilation_ratio)
        face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255

        # Face crop
        r_pad_crop = int((re - rb) * args.facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * args.facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
        face_img, _ = crop_and_pad(face_img, crop_rect)
        face_mask, _ = crop_and_pad(face_mask, crop_rect)
        face_img = cv2.resize(face_img, (args.W, args.H))
        face_mask = cv2.resize(face_mask, (args.W, args.H))

    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0

    with open(log_file, 'a') as f:
        f.write(f'{os.path.basename(audio_path)}:')

    # Generate the video
    pipeline_output = pipe(
        ref_image_pil,
        audio_path,
        face_mask_tensor,
        width,
        height,
        args.L,
        config.steps,
        config.cfg,
        generator=generator,
        audio_sample_rate=args.sample_rate,
        context_frames=args.context_frames,
        fps=final_fps,
        context_overlap=args.context_overlap,
        # faster diffusion
        prior_noise_injection=config.prior_noise_injection,
        cond=config.cond,
        acc=config.acc_mode,
        ############# test efficiency ############
        log_file=log_file,
        log_latency=log_latency,
        log_flops=log_flops,
        return_dict=True,
    )
    
    # Save the raw video
    output_video_path = os.path.join(output_dir, f"{base_output_name}.mp4")
    save_videos_grid(
        pipeline_output.videos,
        output_video_path,
        n_rows=1,
        fps=final_fps,
    )

    # Add audio to the video
    video_with_audio_path = os.path.join(output_dir, f"{base_output_name}_withaudio.mp4")
    video_clip = VideoFileClip(output_video_path)
    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(video_with_audio_path, codec="libx264", audio_codec="aac")

    # Remove the raw video file after creating the version with audio
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
        print(f"Removed raw video file: {output_video_path}")
        os.rename(video_with_audio_path, output_video_path)
        print(f"Renamed video with audio to: {output_video_path}")
        
    print(f"Successfully processed: {base_output_name}")
    print(f"Output saved to: {video_with_audio_path}")

    return True

def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)


    ############# model_init started #############

    ## vae init
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    ## denoising net init
    if os.path.exists(config.motion_module_path):
        ### stage1 + stage2
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        ### only stage1
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
            }
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )

    ## face locator init
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )
    face_locator.load_state_dict(torch.load(config.face_locator_path))

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    ### load face detector params
    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

    ############# model_init finished #############

    width, height = args.W, args.H
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # multi-gpu
    multi_gpu_denoising_unets = None
    gpu_ids = None

    if device == "cuda":
        if hasattr(config, 'parallel_unet_kwargs') and config.parallel_unet_kwargs.gpu_ids:
            gpu_ids = config.parallel_unet_kwargs.gpu_ids
        else:
            # Default to all available GPUs if not specified
            gpu_ids = list(range(torch.cuda.device_count()))
        multi_gpu_denoising_unets = initialize_models_on_gpus(denoising_unet, gpu_ids, weight_dtype)
    
        # Make sure at least one GPU is available
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0

    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
        # multi-gpu
        multi_gpu_denoising_unets=multi_gpu_denoising_unets,
        gpu_ids=gpu_ids,
        max_parallel_per_gpu=config.parallel_unet_kwargs.max_parallel_per_gpu,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    #------------------------------
    # register_parallel_pipeline(pipeline) # 2. enable parallel. If memory is limited, replace it with  `register_normal_pipeline(pipe)`
    register_faster_forward(pipe.denoising_unet)  # 3. encoder propagation
    for idx in pipe.multi_gpu_denoising_unets.keys():
        register_faster_forward(pipe.multi_gpu_denoising_unets[idx])
    #------------------------------

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"
    if args.output_dir:
        save_dir = Path(f"{args.output_dir}_{date_str}_{time_str}")
    else:
        save_dir = Path(f"output/{date_str}/{save_dir_name}")
    
    save_dir.mkdir(exist_ok=True, parents=True)

    # Set up logging
    log_file = save_dir / f"echomimic_inference_log_{time_str}.txt"

    # Batch processing logic
    if args.batch_mode and args.audio_dir and args.image_dir:
        # Process in batch mode
        print(f"Running in batch mode with audio_dir={args.audio_dir}, image_dir={args.image_dir}")
        
        # Get all audio files
        audio_files = []
        for ext in ['wav', 'mp3', 'ogg']:
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, f"*.{ext}")))
        
        # Get all image files
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
        
        # Create dictionaries for quick lookup
        audio_dict = {get_file_basename(f): f for f in audio_files}
        image_dict = {get_file_basename(f): f for f in image_files}
        
        # Find matching pairs (same filename in both directories)
        pairs = []
        for basename in audio_dict:
            if basename in image_dict:
                pairs.append((image_dict[basename], audio_dict[basename], basename))
        
        print(f"Found {len(pairs)} matching (image, audio) pairs")
        
        # Process each pair
        successful = 0
        
        for i, (image_path, audio_path, basename) in enumerate(tqdm(pairs, desc="Processing pairs")):
            print(f"\nProcessing pair {i+1}/{len(pairs)}: {basename}")
            print(f"  Image: {image_path}")
            print(f"  Audio: {audio_path}")
            
            # Set generator for reproducibility
            if args.seed is not None and args.seed > -1:
                generator = torch.manual_seed(args.seed)
            else:
                generator = torch.manual_seed(random.randint(100, 1000000))
            
            # if args.log_latency:
            #     start_event = torch.cuda.Event(enable_timing=True)
            #     end_event = torch.cuda.Event(enable_timing=True)
            #     start_event.record()

            # if args.log_flops:
            #     # Pass a list of the main computational modules instead of the entire pipeline
            #     modules_to_monitor = []
                
            #     potential_modules = [
            #         pipe.denoising_unet,
            #         pipe.reference_unet, 
            #         pipe.vae,
            #         pipe.audio_guider,
            #         pipe.face_locator
            #     ]

            #     for module in potential_modules:
            #         if hasattr(module, 'named_modules'):
            #             modules_to_monitor.append(module)
                
            #     # Add multi-GPU models if they exist
            #     if pipe.multi_gpu_denoising_unets:
            #         for gpu_model in pipe.multi_gpu_denoising_unets.values():
            #             if hasattr(gpu_model, 'named_modules'):
            #                 modules_to_monitor.append(gpu_model)
                
            #     # Skip audio_guider since it's not a proper PyTorch module
            #     flops_counter = FlopCounterMode(modules_to_monitor, depth=None, display=False)
            # else:
            #     flops_counter = nullcontext()
            
            # with flops_counter:
            success = process_single_pair(
                image_path,
                audio_path,
                save_dir,
                pipe,
                face_detector,
                args,
                config,
                generator,
                weight_dtype,
                log_file=log_file,
                log_latency=args.log_latency,
                log_flops=args.log_flops,
            )

            # if args.log_latency:
            #     end_event.record()
            #     torch.cuda.synchronize()
            #     latency = start_event.elapsed_time(end_event)
            #     print(f"{basename}: {latency} ms")

            # if log_file is not None:
            #     assert not (args.log_latency and args.log_flops)
            #     if args.log_latency:
            #         with open(log_file, 'a') as f:
            #             f.write(f'latency: {latency} ms\n')

            #     if args.log_flops:
            #         gflops = flops_counter.get_total_flops() / 1e9
            #         with open(log_file, 'a') as f:
            #             f.write(f'gflops: {gflops} GFLOPs\n')

            if success:
                successful += 1
                print(f"Successfully processed {basename}")
            else:
                print(f"Failed to process {basename}")
        
        print(f"Completed processing {successful}/{len(pairs)} pairs successfully")


        # Final stats
        with open(log_file, 'a') as f:
            f.write(f"\nSummary:\n")
            f.write(f"  Total pairs: {len(pairs)}\n")
            f.write(f"  Successfully processed: {successful}\n")

    else:
        # Original code for processing test cases from config
        for ref_image_path in config["test_cases"].keys():
            base_name = Path(ref_image_path).stem
            # source_fg_mask = load_mask_fg(f'../../data/HDTF/masks/resnet34/{base_name}.npy', device)
            source_fg_mask = load_mask_fg(f'./assets/mask_fg/{base_name}.npy', device)
            for audio_path in config["test_cases"][ref_image_path]:

                if args.seed is not None and args.seed > -1:
                    generator = torch.manual_seed(args.seed)
                else:
                    generator = torch.manual_seed(random.randint(100, 1000000))

                ref_name = Path(ref_image_path).stem
                audio_name = Path(audio_path).stem
                final_fps = args.fps

                #### face musk prepare
                face_img = cv2.imread(ref_image_path)
                face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')

                det_bboxes, probs = face_detector.detect(face_img)
                select_bbox = select_face(det_bboxes, probs)
                # face_mask[:, :] = 255
                if select_bbox is None:
                    face_mask[:, :] = 255
                else:
                    xyxy = select_bbox[:4]
                    xyxy = np.round(xyxy).astype('int')
                    rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
                    r_pad = int((re - rb) * args.facemusk_dilation_ratio)
                    c_pad = int((ce - cb) * args.facemusk_dilation_ratio)
                    face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255

                    #### face crop
                    r_pad_crop = int((re - rb) * args.facecrop_dilation_ratio)
                    c_pad_crop = int((ce - cb) * args.facecrop_dilation_ratio)
                    crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
                    print(crop_rect)
                    face_img, _ = crop_and_pad(face_img, crop_rect)
                    face_mask, _ = crop_and_pad(face_mask, crop_rect)
                    face_img = cv2.resize(face_img, (args.W, args.H))
                    face_mask = cv2.resize(face_mask, (args.W, args.H))

                ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
                face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0

                video = pipe(
                    ref_image_pil,
                    audio_path,
                    face_mask_tensor,
                    width,
                    height,
                    args.L,
                    config.steps,
                    config.cfg,
                    generator=generator,
                    audio_sample_rate=args.sample_rate,
                    context_frames=args.context_frames,
                    fps=final_fps,
                    context_overlap=args.context_overlap,
                    # faster diffusion
                    prior_noise_injection=config.prior_noise_injection,
                    cond=config.cond,
                    acc=config.acc_mode,
                ).videos

                video = video
                save_videos_grid(
                    video,
                    f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
                    n_rows=1,
                    fps=final_fps,
                )

                video_clip = VideoFileClip(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4")
                audio_clip = AudioFileClip(audio_path)
                video_clip = video_clip.set_audio(audio_clip)
                video_clip.write_videofile(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4", codec="libx264", audio_codec="aac")
                print(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4")


if __name__ == "__main__":
    main()
