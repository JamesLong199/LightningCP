# pylint: disable=E1101
# scripts/inference_batch_hdtf.py

"""
This script processes batches of HDTF files with the FaceAnimatePipeline.

It extends the single-file inference script to process multiple audio/image pairs,
supporting efficient batch processing across multiple GPUs.

Usage:
python scripts/inference_batch_hdtf.py --config configs/inference/default.yaml 
    --audio_dir /path/to/audio_files --image_dir /path/to/image_files --output_dir /path/to/output
"""

import argparse
import os
import glob
import logging
import datetime
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm
import copy

import sys
# Add parent directory to path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.config import filter_non_none
from hallo.utils.util import tensor_to_video

from hallo.models.utils_fasterdiffusion import register_faster_forward_multi_gpu as register_faster_forward
from hallo.animate.fasterdiffusion_multi_gpu import FaceAnimatePipeline

# Configure logger
logger = logging.getLogger("hallo")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
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
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }


def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)] for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)
    return audio_emb


def initialize_models_on_gpus(net, gpu_ids, weight_dtype):
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
        gpu_denoising_unet = copy.deepcopy(net.denoising_unet)
        
        # Move to the appropriate GPU
        gpu_denoising_unet.to(device, dtype=weight_dtype)

        # Store in dictionary
        gpu_models[gpu_id] = gpu_denoising_unet
    
    return gpu_models


def get_file_basename(file_path):
    """
    Extract the base filename without extension from a file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Base filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def process_single_pair(
    input_image_path,
    input_audio_path,
    output_dir,
    config,
    net,
    pipeline,
    generator,
    device,
    weight_dtype,
    log_file=None,
    log_latency=False,
    log_flops=False
):
    """
    Process a single image-audio pair to generate an animated video.
    
    Args:
        input_image_path (str): Path to the input source image
        input_audio_path (str): Path to the input audio file
        output_dir (str): Directory to save output
        config: Configuration object
        net: Network model containing all components
        pipeline: FaceAnimatePipeline instance
        generator: Random generator for reproducibility
        device: Torch device
        weight_dtype: Data type for model weights
        log_file: File for logging performance data
        log_latency: Whether to log latency information
        log_flops: Whether to log FLOPs information
        
    Returns:
        str: Path to the output video file
    """
    basename = f"{get_file_basename(input_audio_path)}"
    logger.info(f"Processing pair: {basename}")
    
    # Create subdirectory for this pair's intermediate files
    pair_save_path = os.path.join(output_dir, basename)
    os.makedirs(pair_save_path, exist_ok=True)
    
    # Motion scale settings from config
    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]
    
    # Image size and clip length
    img_size = (config.data.source_image.width, config.data.source_image.height)
    clip_length = config.data.n_sample_frames
    
    # Face analysis model path
    face_analysis_model_path = config.face_analysis.model_path
    
    # Process source image
    logger.info(f"Processing image: {input_image_path}")
    with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(
            input_image_path, pair_save_path, config.face_expand_ratio)
    
    # Process audio
    logger.info(f"Processing audio: {input_audio_path}")
    sample_rate = config.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = config.data.export_video.fps
    wav2vec_model_path = config.wav2vec.model_path
    wav2vec_only_last_features = config.wav2vec.features == "last"
    audio_separator_model_file = config.audio_separator.model_path
    
    with AudioProcessor(
        sample_rate,
        fps,
        wav2vec_model_path,
        wav2vec_only_last_features,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(pair_save_path, "audio_preprocess")
    ) as audio_processor:
        audio_emb, audio_length = audio_processor.preprocess(input_audio_path, clip_length)
    
    # Process audio embeddings
    audio_emb = process_audio_emb(audio_emb)
    
    # Prepare tensors
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
    
    # Calculate how many clips we need to process
    times = audio_emb.shape[0] // clip_length
    tensor_result = []
    
    # Process each clip
    for t in tqdm(range(times), desc=f"Generating video for {basename}"):
        if len(tensor_result) == 0:
            # The first iteration
            motion_zeros = source_image_pixels.repeat(
                config.data.n_motion_frames, 1, 1, 1)
            motion_zeros = motion_zeros.to(
                dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat(
                [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
        else:
            motion_frames = tensor_result[-1][0]
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[0-config.data.n_motion_frames:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(
                dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat(
                [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames
    
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
    
        audio_tensor = audio_emb[
            t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
        ]
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(
            device=net.audioproj.device, dtype=net.audioproj.dtype)
        audio_tensor = net.audioproj(audio_tensor)
    
        pipeline_output = pipeline.call(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            face_emb=source_image_face_emb,
            face_mask=source_image_face_region,
            pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=source_image_face_mask,
            pixel_values_lip_mask=source_image_lip_mask,
            width=img_size[0],
            height=img_size[1],
            video_length=clip_length,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.cfg_scale,
            generator=generator,
            motion_scale=motion_scale,
            # faster diffusion
            prior_noise_injection=config.prior_noise_injection,
            cond=config.cond,
            ############# test efficiency ############
            log_file=log_file,
            log_latency=log_latency,
            log_flops=log_flops,
        )
    
        tensor_result.append(pipeline_output.videos)
    
    # Combine all clips
    tensor_result = torch.cat(tensor_result, dim=2)
    tensor_result = tensor_result.squeeze(0)
    tensor_result = tensor_result[:, :audio_length]
    
    # Create output filename and save video
    output_file = os.path.join(output_dir, f"{basename}.mp4")
    tensor_to_video(tensor_result, output_file, input_audio_path)
    
    logger.info(f"Saved video to: {output_file}")
    return output_file


def main():
    """Main entry point for batch inference"""
    parser = argparse.ArgumentParser()
    
    # Configuration
    parser.add_argument(
        "-c", "--config", default="configs/inference/default.yaml",
        help="Path to configuration file")
    
    # Input directories
    parser.add_argument(
        "--audio_dir", type=str, required=True,
        help="Directory containing audio files")
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing image files")
    
    # Output directory
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save output videos")
    
    # Optional arguments
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility")
    parser.add_argument(
        "--log_latency", action="store_true",
        help="Log latency information")
    parser.add_argument(
        "--log_flops", action="store_true",
        help="Log FLOPS information")
    
    # Parse args
    args = parser.parse_args()
    
    # Load config
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir + f'_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_dir, f"performance_log_{timestamp}.txt")
    
    # Write config to log
    with open(log_file, 'w') as f:
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Inference steps: {config.inference_steps}\n")
        f.write(f"CFG scale: {config.cfg_scale}\n")
        f.write(f"Prior noise injection: {config.prior_noise_injection}\n")
        f.write(f"Weight dtype: {config.weight_dtype}\n")
        f.write("-" * 50 + "\n")
    
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Set dtype
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32
    logger.info(f"Using weight dtype: {weight_dtype}")
    
    # Build models
    logger.info("Building models...")
    
    # Initialize scheduler
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    # Initialize models
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
    
    audio_ckpt_dir = config.audio_ckpt_dir
    
    # Freeze models
    vae.requires_grad_(False)
    image_proj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audio_proj.requires_grad_(False)
    
    reference_unet.enable_gradient_checkpointing()
    denoising_unet.enable_gradient_checkpointing()
    
    # Build network
    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        image_proj,
        audio_proj,
    )
    
    # Load checkpoint
    ckpt_path = os.path.join(audio_ckpt_dir, "net.pth")
    logger.info(f"Loading weights from: {ckpt_path}")
    
    m, u = net.load_state_dict(
        torch.load(ckpt_path, map_location="cpu"),
    )
    assert len(m) == 0 and len(u) == 0, "Failed to load correct checkpoint."
    
    # Set up multi-GPU
    multi_gpu_denoising_unets = None
    gpu_ids = None
    
    if device.type == "cuda":
        if hasattr(config, 'parallel_unet_kwargs') and config.parallel_unet_kwargs.gpu_ids:
            gpu_ids = config.parallel_unet_kwargs.gpu_ids
        else:
            # Default to all available GPUs if not specified
            gpu_ids = list(range(torch.cuda.device_count()))
        multi_gpu_denoising_unets = initialize_models_on_gpus(net, gpu_ids, weight_dtype)
    
        # Make sure at least one GPU is available
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0
    
    # Build pipeline
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=val_noise_scheduler,
        image_proj=net.imageproj,
        # multi-gpu
        multi_gpu_denoising_unets=multi_gpu_denoising_unets,
        gpu_ids=gpu_ids,
        max_parallel_per_gpu=config.parallel_unet_kwargs.max_parallel_per_gpu,
    )
    pipeline.to(device=device, dtype=weight_dtype)
    
    # Register faster forward
    register_faster_forward(pipeline.denoising_unet)
    for idx in pipeline.multi_gpu_denoising_unets.keys():
        register_faster_forward(pipeline.multi_gpu_denoising_unets[idx])
    
    # Find all audio and image files
    audio_files = []
    for ext in ['wav', 'mp3', 'ogg']:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, f"*.{ext}")))
    
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
    
    # Create dictionaries for quick lookup
    audio_dict = {get_file_basename(f): f for f in audio_files}
    image_dict = {get_file_basename(f): f for f in image_files}
    
    # Find matching files by basename
    matching_pairs = []
    
    # First try exact matches (same filename in both directories)
    for basename in audio_dict:
        if basename in image_dict:
            matching_pairs.append((image_dict[basename], audio_dict[basename]))
    
    # If no exact matches, use all combinations
    if not matching_pairs:
        logger.info("No matching filenames found. Processing all combinations.")
        for image_file in image_files:
            for audio_file in audio_files:
                matching_pairs.append((image_file, audio_file))
    
    logger.info(f"Found {len(matching_pairs)} image-audio pairs to process")
    
    # Fixed random seed
    generator = torch.manual_seed(args.seed)
    
    # Process each pair
    successful = 0
    for i, (image_path, audio_path) in enumerate(matching_pairs):
        logger.info(f"\nProcessing pair {i+1}/{len(matching_pairs)}")
        # try:
        output_path = process_single_pair(
            image_path,
            audio_path,
            output_dir,
            config,
            net,
            pipeline,
            generator,
            device,
            weight_dtype,
            log_file,
            args.log_latency,
            args.log_flops
        )
        successful += 1
        logger.info(f"Successfully processed pair {i+1}: {output_path}")
        # except Exception as e:
        #     logger.error(f"Error processing pair {i+1}: {str(e)}")
    
    logger.info(f"Completed processing {successful}/{len(matching_pairs)} pairs successfully")


if __name__ == "__main__":
    main()