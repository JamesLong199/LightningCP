# inference_batch_hdtf_multi_gpu.py
"""
This script processes a batch of (image, audio) pairs for inference using multiple GPUs.
It combines the batch processing capabilities of inference_batch_hdtf.py with
the multi-GPU acceleration from inference_multi_gpu.py.

Usage:
python scripts/inference_batch_hdtf_multi_gpu.py --audio_dir /path/to/audio/files --image_dir /path/to/image/files --mask_dir /path/to/mask/files --output_dir /path/to/output/directory
"""

import argparse
import os
import glob
import sys
import torch
import threading
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import cv2
import copy
import datetime

# Add parent directory to path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

# Import necessary components
from diffusers import AutoencoderKL, DDIMScheduler
from hallo.animate.face_animate_multi_gpu import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d_parallel import UNet3DConditionModel
from hallo.utils.config import filter_non_none
from hallo.utils.util import tensor_to_video
from hallo.models.masking_utils import load_mask_fg


class Net(torch.nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    """
    def __init__(
        self,
        reference_unet,
        denoising_unet,
        face_locator,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self,):
        """Empty function to override abstract function of nn Module"""
        pass

    def get_modules(self):
        """Return all modules"""
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }


def process_audio_emb(audio_emb):
    """Process the audio embedding to concatenate with other tensors."""
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)] for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)
    return audio_emb


def initialize_models_on_gpus(net, gpu_ids, weight_dtype):
    """Initialize denoising_unet on GPUs other than GPU 0."""
    print(f"Initializing denoising_unet on GPUs: {gpu_ids[1:] if len(gpu_ids) > 1 else 'None'}")
    
    # Dictionary to store denoising_unet models for each GPU
    gpu_models = {}
    
    # Skip GPU 0 as all models are already there
    for gpu_id in gpu_ids[1:]:
        device = f"cuda:{gpu_id}"
        print(f"Initializing denoising_unet on {device}")
        
        # Create a deep copy of denoising_unet for this GPU
        gpu_denoising_unet = copy.deepcopy(net.denoising_unet)
        
        # Move to the appropriate GPU with the correct dtype
        gpu_denoising_unet.to(device=device, dtype=weight_dtype)
        
        # Store in dictionary
        gpu_models[gpu_id] = gpu_denoising_unet
    
    return gpu_models


def get_file_basename(file_path):
    """Extract the base filename without extension from a file path."""
    return os.path.splitext(os.path.basename(file_path))[0]


def load_models(config, device, weight_dtype):
    """Load all models needed for inference with multi-GPU support."""
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
    
    # multi-gpu
    multi_gpu_denoising_unets = None
    gpu_ids = None

    if device.type == "cuda":
        if hasattr(config, 'parallel_unet_kwargs') and hasattr(config.parallel_unet_kwargs, 'gpu_ids') and config.parallel_unet_kwargs.gpu_ids:
            gpu_ids = config.parallel_unet_kwargs.gpu_ids
        else:
            # Default to all available GPUs if not specified
            gpu_ids = list(range(torch.cuda.device_count()))
        
        multi_gpu_denoising_unets = initialize_models_on_gpus(net, gpu_ids, weight_dtype)
    
        # Make sure at least one GPU is available
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0
    
    # Create multi-GPU pipeline
    # max_parallel_per_gpu = 3
    # if hasattr(config, 'parallel_unet_kwargs') and hasattr(config.parallel_unet_kwargs, 'max_parallel_per_gpu'):
    max_parallel_per_gpu = config.parallel_unet_kwargs.max_parallel_per_gpu
    
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=val_noise_scheduler,
        image_proj=net.imageproj,
        multi_gpu_denoising_unets=multi_gpu_denoising_unets,
        gpu_ids=gpu_ids,
        max_parallel_per_gpu=max_parallel_per_gpu,
    )
    pipeline.to(device=device, dtype=weight_dtype)
    
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
    
    return pipeline, net, audio_processor, image_processor, gpu_ids


def process_single_pair(
    image_path, 
    audio_path, 
    mask_path,
    output_path, 
    temp_dir,
    config, 
    pipeline, 
    net, 
    audio_processor, 
    image_processor, 
    device, 
    weight_dtype,
    motion_scale,
    log_file,
    log_latency,
    log_flops,
):
    """
    Process a single (image, audio, mask) pair.
    """
    # try:
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
    
    # Load mask
    source_fg_mask = load_mask_fg(mask_path, device)
    
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

        if not config.bg_cache_kwargs.use_masked_attention and config.bg_cache_kwargs.bg_cache_interval <= 1:
            source_fg_mask = None
        
        pipeline_output = pipeline(
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
            ############# DeepCache params ############
            cache_interval=config.deepcache_kwargs.cache_interval, 
            cache_layer_id=config.deepcache_kwargs.cache_layer_id, 
            cache_block_id=config.deepcache_kwargs.cache_block_id, 
            uniform=config.deepcache_kwargs.uniform,
            interval_seq=config.deepcache_kwargs.interval_seq,
            ############ parallel unet ############
            estimate_input_latents=config.parallel_unet_kwargs.estimate_input_latents,
            ############# ours ############
            bg_cache_interval=config.bg_cache_kwargs.bg_cache_interval,
            use_masked_attention=config.bg_cache_kwargs.use_masked_attention,
            masked_layers=config.bg_cache_kwargs.masked_layers,
            remove_ref_list=config.bg_cache_kwargs.remove_ref_list,
            ############# test efficiency ############
            log_file=log_file,
            log_latency=log_latency,
            log_flops=log_flops,
        )
        
        tensor_result.append(pipeline_output.videos)
    
    # Combine results and save video
    tensor_result = torch.cat(tensor_result, dim=2)
    tensor_result = tensor_result.squeeze(0)
    tensor_result = tensor_result[:, :audio_length]
    
    tensor_to_video(tensor_result, output_path, audio_path)

    # denoising_times = pipeline_output.denoising_times
    # avg_denoising_time = sum(denoising_times) / len(denoising_times)

    # with open(log_file, 'a') as f:
    #     f.write(f'{os.path.basename(audio_path)}: {avg_denoising_time}\n')
    return True
    # except Exception as e:
    #     print(f"Error processing {image_path} and {audio_path}: {e}")
    #     return False


def batch_inference(args):
    """Process a batch of (image, audio, mask) pairs for inference using multiple GPUs."""
    # Create output directory if it doesn't exist
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_file = f'logs/denoising_times_hdtf_{current_time}.txt'

    args.output_dir = args.output_dir + f'_{current_time}'
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = f'{args.output_dir}/denoising_times_hdtf_{current_time}.txt'
    # Load configuration
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)

    with open(log_file, 'a') as f:
        f.write(f'Multi-GPU config:\n'
                f'gpu_ids={config.parallel_unet_kwargs.gpu_ids if hasattr(config, "parallel_unet_kwargs") and hasattr(config.parallel_unet_kwargs, "gpu_ids") else "all"}\n'
                f'max_parallel_per_gpu={config.parallel_unet_kwargs.max_parallel_per_gpu}\n'
                f'estimate_input_latents={config.parallel_unet_kwargs.estimate_input_latents}\n')
        f.write(f'DeepCache params:\n'
                f'cache_interval={config.deepcache_kwargs.cache_interval}\n'
                f'cache_layer_id={config.deepcache_kwargs.cache_layer_id}\n'
                f'cache_block_id={config.deepcache_kwargs.cache_block_id}\n'
                f'uniform={config.deepcache_kwargs.uniform}\n'
                f'interval_seq={config.deepcache_kwargs.interval_seq}\n')
        f.write(f'BG Cache params:\n'
                f'bg_cache_interval={config.bg_cache_kwargs.bg_cache_interval}\n'
                f'use_masked_attention={config.bg_cache_kwargs.use_masked_attention}\n'
                f'masked_layers={config.bg_cache_kwargs.masked_layers}\n'
                f'remove_ref_list={config.bg_cache_kwargs.remove_ref_list}\n')
        f.write('-' * 50 + '\n')
    
    # Set device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    print('weight_dtype:', weight_dtype, flush=True)
    
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
        if basename in image_dict and basename in mask_dict:
            pairs.append((image_dict[basename], audio_dict[basename], mask_dict[basename], basename))

    print(f"Found {len(pairs)} matching (image, audio, mask) pairs")
    
    # Load models with multi-GPU support
    pipeline, net, audio_processor, image_processor, gpu_ids = load_models(
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

        success = process_single_pair(
            image_path, 
            audio_path,
            mask_path, 
            output_path, 
            temp_dir,
            config, 
            pipeline, 
            net, 
            audio_processor, 
            image_processor, 
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for multiple (image, audio, mask) pairs using multiple GPUs")
    
    # Required arguments
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Directory containing audio files")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing image files")
    parser.add_argument("--mask_dir", type=str, required=True, 
                        help="Directory containing mask files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save output videos")
    
    # Optional arguments (same as in inference.py)
    parser.add_argument("-c", "--config", default="configs/inference/default.yaml",
                        help="Path to config file")
    parser.add_argument("--pose_weight", type=float, help="Weight of pose", required=False)
    parser.add_argument("--face_weight", type=float, help="Weight of face", required=False)
    parser.add_argument("--lip_weight", type=float, help="Weight of lip", required=False)
    parser.add_argument("--face_expand_ratio", type=float, help="Face region expansion ratio", required=False)
    parser.add_argument("--audio_ckpt_dir", "--checkpoint", type=str, 
                        help="Specific checkpoint directory", required=False)
    parser.add_argument("--log_latency", default=False, help="Log latency")
    parser.add_argument("--log_flops", default=False, help="Log flops")
    
    args = parser.parse_args()
    
    batch_inference(args)