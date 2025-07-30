import argparse
import logging
import os
import glob
import torch
import datetime
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from packaging import version
from tqdm import tqdm

from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d_parallel import UNet3DConditionModel
from memo.pipelines.video_pipeline_multi_gpu import VideoPipeline
from memo.utils.audio_utils import extract_audio_emotion_labels, load_emotion_models, preprocess_audio, resample_audio
from memo.utils.vision_utils import preprocess_image, tensor_to_video
from memo.models.masking_utils import load_mask_fg
import copy

logger = logging.getLogger("memo")
logger.setLevel(logging.INFO)

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
    parser = argparse.ArgumentParser(description="Batch inference script for MEMO with multi-GPU support")

    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output videos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_latency", default=False, help="Log latency")
    parser.add_argument("--log_flops", default=False, help="Log flops")
    return parser.parse_args()

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
        gpu_denoising_unet.to(device, dtype=weight_dtype)

        # Store in dictionary
        gpu_models[gpu_id] = gpu_denoising_unet
    
    return gpu_models

def process_single_pair(
    input_image_path,
    input_audio_path,
    mask_path,
    output_dir,
    config,
    models,
    generator,
    device,
    weight_dtype,
    log_file,
    log_latency,
    log_flops,
    emotion_models=None,
):
    # Extract models
    vae, reference_net, diffusion_net, image_proj, audio_proj, pipeline = models
    
    # Create output paths
    basename = f"{get_file_basename(input_image_path)}_{get_file_basename(input_audio_path)}"
    output_video_path = os.path.join(output_dir, f"{basename}.mp4")
    cache_dir = os.path.join(output_dir, "audio_preprocess", basename)
    os.makedirs(cache_dir, exist_ok=True)

    # Load mask
    source_fg_mask = load_mask_fg(mask_path, device)

    logger.info(f"Processing image {input_image_path}")
    img_size = (config.resolution, config.resolution)
    pixel_values, face_emb = preprocess_image(
        face_analysis_model=os.path.join(config.misc_model_dir, "misc/face_analysis"),
        image_path=input_image_path,
        image_size=config.resolution,
    )

    logger.info(f"Processing audio {input_audio_path}")
    resampled_audio_path = resample_audio(
        input_audio_path,
        os.path.join(cache_dir, f"{os.path.basename(input_audio_path).split('.')[0]}-16k.wav"),
    )
    
    vocal_separator = os.path.join(config.misc_model_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")
    audio_emb, audio_length = preprocess_audio(
        wav_path=resampled_audio_path,
        num_generated_frames_per_clip=config.num_generated_frames_per_clip,
        fps=config.fps,
        wav2vec_model=config.wav2vec,
        vocal_separator_model=vocal_separator,
        cache_dir=cache_dir,
        device=device,
    )

    logger.info("Processing audio emotion")
    audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
        model="memoavatar/memo",
        wav_path=resampled_audio_path,
        emotion2vec_model=config.emotion2vec,
        audio_length=audio_length,
        device=device,
        pre_loaded_models=emotion_models,
    )

    video_frames = []
    num_clips = audio_emb.shape[0] // config.num_generated_frames_per_clip
    
    denoising_times = []
    for t in tqdm(range(num_clips), desc=f"Generating video for {basename}"):
        if len(video_frames) == 0:
            # Initialize the first past frames with reference image
            past_frames = pixel_values.repeat(config.num_init_past_frames, 1, 1, 1)
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
        else:
            past_frames = video_frames[-1][0]
            past_frames = past_frames.permute(1, 0, 2, 3)
            past_frames = past_frames[0 - config.num_past_frames :]
            past_frames = past_frames * 2.0 - 1.0
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        audio_tensor = (
            audio_emb[
                t * config.num_generated_frames_per_clip : min(
                    (t + 1) * config.num_generated_frames_per_clip, audio_emb.shape[0]
                )
            ]
            .unsqueeze(0)
            .to(device=audio_proj.device, dtype=audio_proj.dtype)
        )
        audio_tensor = audio_proj(audio_tensor)

        audio_emotion_tensor = audio_emotion[
            t * config.num_generated_frames_per_clip : min(
                (t + 1) * config.num_generated_frames_per_clip, audio_emb.shape[0]
            )
        ]

        if not config.bg_cache_kwargs.use_masked_attention and config.bg_cache_kwargs.bg_cache_interval <= 1:
            source_fg_mask = None

        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            audio_emotion=audio_emotion_tensor,
            emotion_class_num=num_emotion_classes,
            face_emb=face_emb,
            width=img_size[0],
            height=img_size[1],
            video_length=config.num_generated_frames_per_clip,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.cfg_scale,
            generator=generator,
            is_new_audio=t == 0,
            ############# DeepCache params ############
            cache_interval=config.deepcache_kwargs.cache_interval, 
            cache_layer_id=config.deepcache_kwargs.cache_layer_id, 
            cache_block_id=config.deepcache_kwargs.cache_block_id,
            uniform=config.deepcache_kwargs.uniform,
            ############ parallel unet ############
            estimate_input_latents=config.parallel_unet_kwargs.estimate_input_latents,
            ############ ours ############
            mask_fg=source_fg_mask,
            bg_cache_interval=config.bg_cache_kwargs.bg_cache_interval,
            use_masked_attention=config.bg_cache_kwargs.use_masked_attention,
            masked_layers=config.bg_cache_kwargs.masked_layers,
            remove_ref_list=config.bg_cache_kwargs.remove_ref_list,
            save_dir=None,
            save_layers=None,
            save_types=None,
            # acc_mode=config.acc_mode,
            ############# test efficiency ############
            log_file=log_file,
            log_latency=log_latency,
            log_flops=log_flops,
        )

        video_frames.append(pipeline_output.videos)
        if hasattr(pipeline_output, 'denoising_times'):
            denoising_times.extend(pipeline_output.denoising_times)

    video_frames = torch.cat(video_frames, dim=2)
    video_frames = video_frames.squeeze(0)
    video_frames = video_frames[:, :audio_length]

    tensor_to_video(video_frames, output_video_path, resampled_audio_path, fps=config.fps)

    # Log performance data
    if denoising_times:
        avg_denoising_time = sum(denoising_times) / len(denoising_times)
        with open(log_file, 'a') as f:
            f.write(f'{basename}: {avg_denoising_time}\n')

    return True

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging and output directories
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_file = f'logs/memo_denoising_times_multi_gpu_{current_time}.txt'
    # os.makedirs('logs', exist_ok=True)
    
    output_dir = args.output_dir + f'_{current_time}'
    os.makedirs(output_dir, exist_ok=True)
    log_file = f'{output_dir}/memo_denoising_times_multi_gpu_{current_time}.txt'
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)
    
    # Set up device and weight dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32
    logger.info(f"Inference dtype: {weight_dtype}")

    # Download face analysis and vocal separator models, if they do not exist
    face_analysis = os.path.join(config.misc_model_dir, "misc/face_analysis")
    os.makedirs(face_analysis, exist_ok=True)
    for model in [
        "1k3d68.onnx",
        "2d106det.onnx",
        "face_landmarker_v2_with_blendshapes.task",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]:
        model_path = os.path.join(face_analysis, "models", model)
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model} to {face_analysis}/models")
            os.system(f"wget -P {face_analysis}/models https://hf-mirror.com/memoavatar/memo/resolve/main/misc/face_analysis/models/{model}")
            # Check if the download was successful
            if not os.path.exists(model_path):
                raise RuntimeError(f"Failed to download {model} to {model_path}")
            # File size check
            if os.path.getsize(model_path) < 1024 * 1024:
                raise RuntimeError(f"{model_path} file seems incorrect (too small), delete it and retry.")
    logger.info(f"Use face analysis models from {face_analysis}")

    vocal_separator = os.path.join(config.misc_model_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")
    if os.path.exists(vocal_separator):
        logger.info(f"Vocal separator {vocal_separator} already exists. Skipping download.")
    else:
        logger.info(f"Downloading vocal separator to {vocal_separator}")
        os.makedirs(os.path.dirname(vocal_separator), exist_ok=True)
        os.system(
            f"wget -P {os.path.dirname(vocal_separator)} https://hf-mirror.com/memoavatar/memo/resolve/main/misc/vocal_separator/Kim_Vocal_2.onnx"
        )

    # Load models (only once)
    logger.info("Loading models")
    vae = AutoencoderKL.from_pretrained(config.vae).to(device=device, dtype=weight_dtype)
    reference_net = UNet2DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="reference_net", use_safetensors=True
    )
    diffusion_net = UNet3DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="diffusion_net", use_safetensors=True
    )
    image_proj = ImageProjModel.from_pretrained(
        config.model_name_or_path, subfolder="image_proj", use_safetensors=True
    )
    audio_proj = AudioProjModel.from_pretrained(
        config.model_name_or_path, subfolder="audio_proj", use_safetensors=True
    )

    vae.requires_grad_(False).eval()
    reference_net.requires_grad_(False).eval()
    diffusion_net.requires_grad_(False).eval()
    image_proj.requires_grad_(False).eval()
    audio_proj.requires_grad_(False).eval()

    # Enable memory-efficient attention for xFormers
    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.info(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            reference_net.enable_xformers_memory_efficient_attention()
            diffusion_net.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Set up multi-GPU
    multi_gpu_denoising_unets = None
    gpu_ids = None

    if device.type == "cuda":
        if hasattr(config, 'parallel_unet_kwargs') and config.parallel_unet_kwargs.gpu_ids:
            gpu_ids = config.parallel_unet_kwargs.gpu_ids
        else:
            # Default to all available GPUs if not specified
            gpu_ids = list(range(torch.cuda.device_count()))
        multi_gpu_denoising_unets = initialize_models_on_gpus(diffusion_net, gpu_ids, weight_dtype)
    
        # Make sure at least one GPU is available
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0

    # Create inference pipeline
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline = VideoPipeline(
        vae=vae,
        reference_net=reference_net,
        diffusion_net=diffusion_net,
        scheduler=noise_scheduler,
        image_proj=image_proj,
        multi_gpu_denoising_unets=multi_gpu_denoising_unets,
        gpu_ids=gpu_ids,
        max_parallel_per_gpu=config.parallel_unet_kwargs.max_parallel_per_gpu,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    # Group models
    models = (vae, reference_net, diffusion_net, image_proj, audio_proj, pipeline)
    
    # Fixed random seed
    generator = torch.manual_seed(args.seed)
    
    # Write configuration to log file
    with open(log_file, 'a') as f:
        f.write(f'Multi-GPU IDs: {gpu_ids}\n')
        f.write(f'DeepCache params:\n'
                f'cache_interval={config.deepcache_kwargs.cache_interval}\n'
                f'cache_layer_id={config.deepcache_kwargs.cache_layer_id}\n'
                f'cache_block_id={config.deepcache_kwargs.cache_block_id}\n'
                f'uniform={config.deepcache_kwargs.uniform}\n'
                # f'interval_seq={config.deepcache_kwargs.interval_seq}\n'
                )
        f.write(f'BG Cache params:\n'
                f'bg_cache_interval={config.bg_cache_kwargs.bg_cache_interval}\n'
                f'use_masked_attention={config.bg_cache_kwargs.use_masked_attention}\n'
                f'masked_layers={config.bg_cache_kwargs.masked_layers}\n'
                f'remove_ref_list={config.bg_cache_kwargs.remove_ref_list}\n')
        f.write('-' * 50 + '\n')
    
    # Load emotion models once for all audio samples
    logger.info("Loading emotion models for batch processing")
    emotion_models = load_emotion_models(
        model="memoavatar/memo",
        emotion2vec_model=config.emotion2vec,
        device=device,
    )
    
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
    
    # Find matching pairs (same filename in both directories)
    pairs = []
    for basename in audio_dict:
        if basename in image_dict and basename in mask_dict:
            pairs.append((image_dict[basename], audio_dict[basename], mask_dict[basename], basename))
    
    logger.info(f"Found {len(pairs)} matching (image, audio, mask) pairs")
    
    # Process each pair
    successful = 0
    for i, (image_path, audio_path, mask_path, basename) in enumerate(tqdm(pairs, desc="Processing pairs")):
        logger.info(f"\nProcessing pair {i+1}/{len(pairs)}: {basename}")
        logger.info(f"  Image: {image_path}")
        logger.info(f"  Audio: {audio_path}")
        logger.info(f"  Mask: {mask_path}")
        
        # try:
        success = process_single_pair(
            image_path, 
            audio_path, 
            mask_path,
            output_dir, 
            config, 
            models, 
            generator, 
            device, 
            weight_dtype,
            log_file,
            args.log_latency,
            args.log_flops,
            emotion_models,
        )
        
        if success:
            successful += 1
            logger.info(f"Successfully processed {basename}")
        else:
            logger.error(f"Failed to process {basename}")
        # except Exception as e:
        #     logger.error(f"Error processing {basename}: {str(e)}")
    
    logger.info(f"Completed processing {successful}/{len(pairs)} pairs successfully")


if __name__ == "__main__":
    main()
