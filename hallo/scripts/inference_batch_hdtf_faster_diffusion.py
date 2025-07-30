# pylint: disable=E1101
"""
This script processes a batch of (image, audio) pairs for inference using faster diffusion optimization.
It takes a directory containing audio files and a directory containing image files,
matches them by filename, and runs the inference process on each pair with faster diffusion techniques.

Usage:
python scripts/inference_batch_fasterdiffusion.py --audio_dir /path/to/audio/files --image_dir /path/to/image/files --output_dir /path/to/output/directory
"""

import argparse
import os
import glob
import sys
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import datetime

# Add parent directory to path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

# Import necessary components
from diffusers import AutoencoderKL, DDIMScheduler
from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.config import filter_non_none
from hallo.utils.util import tensor_to_video
from scripts.inference import Net, process_audio_emb
from hallo.models.masking_utils import load_mask_fg
from hallo.models.utils_fasterdiffusion import register_parallel_pipeline, register_faster_forward

def get_file_basename(file_path):
    """
    Extract the base filename without extension from a file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Base filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def load_models(config, device, weight_dtype):
    """
    Load all models needed for inference.
    
    Args:
        config: Configuration object
        device: Device to load models on
        weight_dtype: Data type for model weights
        
    Returns:
        tuple: (pipeline, net, audio_processor, image_processor)
    """
    print("Loading models...")
    
    # Load schedulers
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    # Load models
    vae = AutoencoderKL.from_pretrained(config.vae.model_path)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.base_model_path, subfolder="unet")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs),
        use_landmark=False,
    )
    face_locator = FaceLocator(conditioning_embedding_channels=320)
    image_proj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    )
    audio_proj = AudioProjModel(
        seq_len=5,
        blocks=12,  # use 12 layers' hidden states of wav2vec
        channels=768,  # audio embedding channel
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device=device, dtype=weight_dtype)

    # Freeze models
    vae.requires_grad_(False)
    image_proj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audio_proj.requires_grad_(False)

    reference_unet.enable_gradient_checkpointing()
    denoising_unet.enable_gradient_checkpointing()

    # Create network
    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        image_proj,
        audio_proj,
    )

    # Load checkpoint
    m, u = net.load_state_dict(
        torch.load(
            os.path.join(config.audio_ckpt_dir, "net.pth"),
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Failed to load correct checkpoint."
    print("Loaded weight from", os.path.join(config.audio_ckpt_dir, "net.pth"))

    # Create pipeline
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=val_noise_scheduler,
        image_proj=net.imageproj,
    )
    pipeline.to(device=device, dtype=weight_dtype)
    
    # Apply faster diffusion optimizations
    register_parallel_pipeline(pipeline)  # Enable parallel processing
    register_faster_forward(pipeline.denoising_unet)  # Enable faster forward pass
    
    # Load processors
    img_size = (config.data.source_image.width, config.data.source_image.height)
    face_analysis_model_path = config.face_analysis.model_path
    image_processor = ImageProcessor(img_size, face_analysis_model_path)
    
    sample_rate = config.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = config.data.export_video.fps
    wav2vec_model_path = config.wav2vec.model_path
    wav2vec_only_last_features = config.wav2vec.features == "last"
    audio_separator_model_file = config.audio_separator.model_path
    
    # Create a default temp directory for the audio processor
    default_temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_audio_processing")
    os.makedirs(default_temp_dir, exist_ok=True)
    
    audio_processor = AudioProcessor(
        sample_rate,
        fps,
        wav2vec_model_path,
        wav2vec_only_last_features,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        default_temp_dir
    )
    
    return pipeline, net, audio_processor, image_processor

def process_single_pair(
    image_path, 
    audio_path, 
    output_path, 
    temp_dir,
    config, 
    pipeline, 
    net, 
    audio_processor, 
    image_processor, 
    source_fg_mask,
    device, 
    weight_dtype,
    motion_scale,
    log_file,
    log_latency,
    log_flops,
):
    """
    Process a single (image, audio) pair.
    
    Args:
        image_path: Path to the source image
        audio_path: Path to the driving audio
        output_path: Path to save the output video
        temp_dir: Directory for temporary files
        config: Configuration object
        pipeline: FaceAnimatePipeline instance
        net: Network instance
        audio_processor: AudioProcessor instance
        image_processor: ImageProcessor instance
        source_fg_mask: Foreground mask dictionary
        device: Device to run inference on
        weight_dtype: Data type for model weights
        motion_scale: List of motion scale values
        log_file: Path to the log file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        audio_processor.temp_dir = os.path.join(temp_dir, "audio_preprocess")
        
        # Process image
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(
            image_path, temp_dir, config.face_expand_ratio)
        
        # Process audio
        clip_length = config.data.n_sample_frames
        audio_emb, audio_length = audio_processor.preprocess(audio_path, clip_length)
        audio_emb = process_audio_emb(audio_emb)
        
        # Prepare image tensors
        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)
        
        source_image_full_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_full_mask
        ]
        source_image_face_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_face_mask
        ]
        source_image_lip_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_lip_mask
        ]
        
        # Run inference
        times = audio_emb.shape[0] // clip_length
        tensor_result = []
        generator = torch.manual_seed(42)
        
        denoising_times = []
        
        for t in range(times):
            print(f"[{t+1}/{times}]")
            
            if len(tensor_result) == 0:
                # The first iteration
                motion_zeros = source_image_pixels.repeat(
                    config.data.n_motion_frames, 1, 1, 1)
                motion_zeros = motion_zeros.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_zeros], dim=0)
            else:
                motion_frames = tensor_result[-1][0]
                motion_frames = motion_frames.permute(1, 0, 2, 3)
                motion_frames = motion_frames[0-config.data.n_motion_frames:]
                motion_frames = motion_frames * 2.0 - 1.0
                motion_frames = motion_frames.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_frames], dim=0)
            
            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
            
            audio_tensor = audio_emb[
                t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
            ]
            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(
                device=net.audioproj.device, dtype=net.audioproj.dtype)
            audio_tensor = net.audioproj(audio_tensor)
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            pipeline_output = pipeline.call(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=source_image_face_emb,
                face_mask=source_image_face_region,
                pixel_values_full_mask=source_image_full_mask,
                pixel_values_face_mask=source_image_face_mask,
                pixel_values_lip_mask=source_image_lip_mask,
                mask_fg=source_fg_mask,
                width=img_size[0],
                height=img_size[1],
                video_length=clip_length,
                num_inference_steps=config.inference_steps,
                guidance_scale=config.cfg_scale,
                generator=generator,
                motion_scale=motion_scale,
                prior_noise_injection=config.prior_noise_injection if hasattr(config, 'prior_noise_injection') else 0.0,
                cond=config.cond if hasattr(config, 'cond') else True,
                max_parallel_per_gpu=config.max_parallel_per_gpu if hasattr(config, 'max_parallel_per_gpu') else 4,
                ############# test efficiency ############
                log_file=log_file,
                log_latency=log_latency,
                log_flops=log_flops,            
            )
            
            end_time.record()
            torch.cuda.synchronize()
            
            inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            denoising_times.append(inference_time)
            
            tensor_result.append(pipeline_output.videos)
        
        # Combine results and save video
        tensor_result = torch.cat(tensor_result, dim=2)
        tensor_result = tensor_result.squeeze(0)
        tensor_result = tensor_result[:, :audio_length]
        
        tensor_to_video(tensor_result, output_path, audio_path)
        
        avg_denoising_time = sum(denoising_times) / len(denoising_times)
        
        with open(log_file, 'a') as f:
            f.write(f'{os.path.basename(audio_path)}: {avg_denoising_time:.4f}s\n')
        
        return True
    except Exception as e:
        print(f"Error processing {image_path} and {audio_path}: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f'ERROR {os.path.basename(audio_path)}: {str(e)}\n')
        return False

def batch_inference(args):
    """
    Process a batch of (image, audio) pairs for inference.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # Create timestamp for output directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{current_time}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = f'{args.output_dir}/inference_times_{current_time}.txt'
    
    # Load configuration
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    
    # Log configuration
    with open(log_file, 'a') as f:
        f.write(f'Faster Diffusion Batch Inference\n')
        f.write(f'Date: {current_time}\n')
        f.write(f'Config: {args.config}\n')
        f.write(f'Audio dir: {args.audio_dir}\n')
        f.write(f'Image dir: {args.image_dir}\n')
        f.write(f'Mask dir: {args.mask_dir}\n')
        f.write(f'Output dir: {args.output_dir}\n')
        f.write('-' * 50 + '\n')
    
    # Set device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    print('Weight dtype:', weight_dtype, flush=True)
    
    # Set motion scale
    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]
    
    # Get all audio files
    audio_files = []
    for ext in ['wav', 'mp3', 'ogg']:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, f"*.{ext}")))
    
    # Get all mask files
    mask_files = []
    for ext in ['npy']:
        mask_files.extend(glob.glob(os.path.join(args.mask_dir, f"*.{ext}")))
    
    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
    
    # Create dictionaries for quick lookup
    audio_dict = {get_file_basename(f): f for f in audio_files}
    image_dict = {get_file_basename(f): f for f in image_files}
    mask_dict = {get_file_basename(f): f for f in mask_files}
    
    # Find matching pairs
    pairs = []
    for basename in audio_dict:
        if basename in image_dict:
            if basename in mask_dict:
                pairs.append((image_dict[basename], audio_dict[basename], mask_dict[basename], basename))
            else:
                print(f"Warning: No mask found for {basename}, skipping")
    
    print(f"Found {len(pairs)} matching (image, audio, mask) pairs")
    
    # Load models once
    pipeline, net, audio_processor, image_processor = load_models(
        config, device, weight_dtype)
    
    # Set image size
    global img_size
    img_size = (config.data.source_image.width, config.data.source_image.height)
    
    # Process each pair
    successful = 0
    for i, (image_path, audio_path, mask_path, basename) in enumerate(tqdm(pairs, desc="Processing pairs")):
        output_path = os.path.join(args.output_dir, f"{basename}.mp4")
        temp_dir = os.path.join(args.output_dir, "temp", basename)
        
        print(f"\nProcessing pair {i+1}/{len(pairs)}: {basename}")
        print(f"  Image: {image_path}")
        print(f"  Audio: {audio_path}")
        print(f"  Mask: {mask_path}")
        print(f"  Output: {output_path}")
        
        # Load mask if available
        source_fg_mask = load_mask_fg(mask_path, device) if mask_path else None

        success = process_single_pair(
            image_path, 
            audio_path, 
            output_path, 
            temp_dir,
            config, 
            pipeline, 
            net, 
            audio_processor, 
            image_processor, 
            source_fg_mask,
            device, 
            weight_dtype,
            motion_scale,
            log_file,
            args.log_latency,
            args.log_flops,
        )
        
        if success:
            successful += 1
            print(f"Successfully processed {basename}")
        else:
            print(f"Failed to process {basename}")
    
    # Clean up
    image_processor.close()
    audio_processor.close()
    
    print(f"Completed processing {successful}/{len(pairs)} pairs successfully")
    print(f"Results saved to {args.output_dir}")
    print(f"Logs saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for multiple (image, audio) pairs with faster diffusion")
    
    # Required arguments
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Directory containing audio files")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing image files")
    parser.add_argument("--mask_dir", type=str, required=True, 
                        help="Directory containing mask files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save output videos")
    
    # Optional arguments
    parser.add_argument("-c", "--config", default="configs/inference/default.yaml",
                        help="Path to config file")
    parser.add_argument("--pose_weight", type=float, help="Weight of pose", required=False)
    parser.add_argument("--face_weight", type=float, help="Weight of face", required=False)
    parser.add_argument("--lip_weight", type=float, help="Weight of lip", required=False)
    parser.add_argument("--face_expand_ratio", type=float, help="Face region expansion ratio", required=False)
    parser.add_argument("--audio_ckpt_dir", "--checkpoint", type=str, 
                        help="Specific checkpoint directory", required=False)
    parser.add_argument("--prior_noise_injection", type=float, 
                        help="Prior noise injection strength", required=False)
    parser.add_argument("--max_parallel_per_gpu", type=int, 
                        help="Maximum number of parallel videos per GPU", required=False)

    parser.add_argument("--log_latency", default=False, help="Log latency")
    parser.add_argument("--log_flops", default=False, help="Log flops")


    args = parser.parse_args()
    
    batch_inference(args)