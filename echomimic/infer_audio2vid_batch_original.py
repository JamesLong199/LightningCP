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
import glob
import random
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
# from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN

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

def get_file_basename(file_path):
    """
    Extract the base filename without extension from a file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Base filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]

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

    # parser.add_argument("--cfg", type=float, default=2.5)
    # parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Add new arguments for batch processing
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--image_dir", type=str, help="Directory containing image files")
    parser.add_argument("--output_dir", type=str, help="Directory to save output videos")
    parser.add_argument("--batch_mode", action="store_true", help="Process files in batch mode")

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

def process_single_pair(
    ref_image_path,
    audio_path,
    output_dir,
    models,
    args,
    config,
    face_detector,
    generator,
):
    # Extract models
    vae, reference_unet, denoising_unet, audio_processor, face_locator, pipe = models
    
    ref_name = Path(ref_image_path).stem
    audio_name = Path(audio_path).stem
    time_str = datetime.now().strftime("%H%M%S")
    final_fps = args.fps
    
    width, height = args.W, args.H
    
    # Create base output filename
    # base_output_name = f"{ref_name}_{audio_name}_{args.H}x{args.W}_{int(config.cfg)}"
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
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=torch.float16 if args.device == "cuda" else torch.float32, device=args.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0

    # Generate the video
    start_time = time.time()
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
        context_overlap=args.context_overlap
    ).videos
    inference_time = time.time() - start_time
    
    # Save the raw video
    output_video_path = os.path.join(output_dir, f"{base_output_name}.mp4")
    save_videos_grid(
        video,
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
    
    print(f"Successfully processed: {base_output_name}")
    print(f"Output saved to: {video_with_audio_path}")
    print(f"Inference time: {inference_time:.2f} seconds")
    
    return inference_time

def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.acc_mode:
        print("Using acc pipeline")
        from src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
    else:
        print("Using normal pipeline")
        from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    # Set up output directory
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"
    
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = Path(f"output/{date_str}/{save_dir_name}")
    
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    log_file = save_dir / f"echomimic_inference_log_{time_str}.txt"
    
    ############# model_init started #############
    print("Loading models...")
    
    # vae init
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to(device, dtype=weight_dtype)

    # reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    # denoising net init
    if os.path.exists(config.motion_module_path):
        # stage1 + stage2
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        # only stage1
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

    # face locator init
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )
    face_locator.load_state_dict(torch.load(config.face_locator_path))

    # load audio processor
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    # load face detector
    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

    ############# model_init finished #############

    # Initialize pipeline
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)
    
    # Group models for easier passing to process function
    models = (vae, reference_unet, denoising_unet, audio_processor, face_locator, pipe)
    
    # Fixed seed if specified
    if args.seed is not None and args.seed > -1:
        generator = torch.manual_seed(args.seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))
    
    # Write configuration to log file
    with open(log_file, 'w') as f:
        f.write(f"EchoMimic Inference Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Width: {args.W}\n")
        f.write(f"  Height: {args.H}\n")
        f.write(f"  Length: {args.L}\n")
        f.write(f"  Seed: {args.seed}\n")
        f.write(f"  CFG Scale: {config.cfg}\n")
        f.write(f"  Steps: {config.steps}\n")
        f.write(f"  FPS: {args.fps}\n")
        f.write(f"  Context Frames: {args.context_frames}\n")
        f.write(f"  Context Overlap: {args.context_overlap}\n")
        f.write(f"  Face Mask Dilation: {args.facemusk_dilation_ratio}\n")
        f.write(f"  Face Crop Dilation: {args.facecrop_dilation_ratio}\n")
        f.write(f"-" * 50 + "\n")
    
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
        total_time = 0.0
        
        for i, (image_path, audio_path, basename) in enumerate(tqdm(pairs, desc="Processing pairs")):
            print(f"\nProcessing pair {i+1}/{len(pairs)}: {basename}")
            print(f"  Image: {image_path}")
            print(f"  Audio: {audio_path}")
            
            try:
                inference_time = process_single_pair(
                    image_path,
                    audio_path,
                    save_dir,
                    models,
                    args,
                    config,
                    face_detector,
                    generator,
                )
                
                successful += 1
                total_time += inference_time
                
                # Write success to log
                with open(log_file, 'a') as f:
                    f.write(f"Processed {basename}: {inference_time:.2f}s\n")
                
            except Exception as e:
                print(f"Error processing {basename}: {str(e)}")
                # Write error to log
                with open(log_file, 'a') as f:
                    f.write(f"Error processing {basename}: {str(e)}\n")
        
        print(f"Completed processing {successful}/{len(pairs)} pairs successfully")
        print(f"Average inference time: {total_time/max(1, successful):.2f} seconds")
        
        # Final stats
        with open(log_file, 'a') as f:
            f.write(f"\nSummary:\n")
            f.write(f"  Total pairs: {len(pairs)}\n")
            f.write(f"  Successfully processed: {successful}\n")
            f.write(f"  Total inference time: {total_time:.2f}s\n")
            f.write(f"  Average inference time: {total_time/max(1, successful):.2f}s\n")
    
    else:
        # Process specified test cases from config
        for ref_image_path in config["test_cases"].keys():
            for audio_path in config["test_cases"][ref_image_path]:
                process_single_pair(
                    ref_image_path,
                    audio_path,
                    save_dir,
                    models,
                    args,
                    face_detector,
                    generator,
                )

if __name__ == "__main__":
    main()