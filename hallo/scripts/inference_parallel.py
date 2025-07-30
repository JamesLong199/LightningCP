# pylint: disable=E1101
# scripts/inference.py

"""
This script contains the main inference pipeline for processing audio and image inputs to generate a video output.

The script imports necessary packages and classes, defines a neural network model, 
and contains functions for processing audio embeddings and performing inference.

The main inference process is outlined in the following steps:
1. Initialize the configuration.
2. Set up runtime variables.
3. Prepare the input data for inference (source image, face mask, and face embeddings).
4. Process the audio embeddings.
5. Build and freeze the model and scheduler.
6. Run the inference loop and save the result.

Usage:
This script can be run from the command line with the following arguments:
- audio_path: Path to the audio file.
- image_path: Path to the source image.
- face_mask_path: Path to the face mask image.
- face_emb_path: Path to the face embeddings file.
- output_path: Path to save the output video.

Example:
python scripts/inference.py --audio_path audio.wav --image_path image.jpg 
    --face_mask_path face_mask.png --face_emb_path face_emb.pt --output_path output.mp4
"""

import argparse
import os

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
import cv2
import numpy as np

import sys
# Step 1: Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Step 2: Append the parent directory to sys.path
sys.path.append(parent_directory)

from hallo.animate.face_animate_parallel import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d_parallel import UNet3DConditionModel
from hallo.utils.config import filter_non_none
from hallo.utils.util import tensor_to_video

# from hallo.models.tome_patch import apply_patch, remove_patch

class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    
    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
        face_locator (FaceLocator): The FaceLocator model used to locate the face in the input image.
        imageproj (nn.Module): The ImageProjector model used to project the source image onto the face.
        audioproj (nn.Module): The AudioProjector model used to project the audio embeddings onto the face.
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

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb



def inference_process(args: argparse.Namespace):
    """
    Perform inference processing.

    Args:
        args (argparse.Namespace): Command-line arguments.

    This function initializes the configuration for the inference process. It sets up the necessary
    modules and variables to prepare for the upcoming inference steps.
    """

    # 1. init config
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    source_image_path = config.source_image
    driving_audio_path = config.driving_audio
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]

    # 2. runtime variables
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    # 3. prepare inference data
    # 3.1 prepare source image, face mask, face embeddings
    img_size = (config.data.source_image.width,
                config.data.source_image.height)
    clip_length = config.data.n_sample_frames
    face_analysis_model_path = config.face_analysis.model_path
    with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(
            source_image_path, save_path, config.face_expand_ratio)

    fg_mask = np.load('examples/mask_fg/1.npy').astype(np.uint8)
    fg_mask_64 = cv2.resize(fg_mask, (64, 64)).flatten()
    bool_fg_mask_64 = torch.from_numpy(fg_mask_64).bool()
    bool_bg_mask_64 = ~bool_fg_mask_64
    bool_fg_mask_64_encoder = torch.cat([bool_fg_mask_64, bool_fg_mask_64], dim=0)
    bool_bg_mask_64_encoder = torch.cat([bool_bg_mask_64, bool_bg_mask_64], dim=0)

    fg_indices_64 = torch.nonzero(bool_fg_mask_64).squeeze()
    bg_indices_64 = torch.nonzero(bool_bg_mask_64).squeeze()
    fg_encoder_indices_64 = torch.nonzero(bool_fg_mask_64_encoder).squeeze()
    bg_encoder_indices_64 = torch.nonzero(bool_bg_mask_64_encoder).squeeze()
    
    # Store indices directly in the dictionary
    source_fg_mask = {
        64: {
            "fg": fg_indices_64,
            "bg": bg_indices_64,
            "fg_encoder": fg_encoder_indices_64,
            "bg_encoder": bg_encoder_indices_64,
        },
    }
    
    # Move indices to the appropriate device
    for size in source_fg_mask:
        for key in source_fg_mask[size]:
            source_fg_mask[size][key] = source_fg_mask[size][key].to(device=device)

    # 3.2 prepare audio embeddings
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
        os.path.join(save_path, "audio_preprocess")
    ) as audio_processor:
        audio_emb, audio_length = audio_processor.preprocess(driving_audio_path, clip_length)

    print('inference 269 audio_emb.shape:', audio_emb.shape, flush=True) # 192, 12, 768

    # 4. build modules
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

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
    # apply_patch(denoising_unet, ratio=0.5)
    # for module_name, module in denoising_unet.named_modules():
    #     print(f"module_name: {module_name}", flush=True)
    # assert 0
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


    # Freeze
    vae.requires_grad_(False)
    image_proj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audio_proj.requires_grad_(False)

    reference_unet.enable_gradient_checkpointing()
    denoising_unet.enable_gradient_checkpointing()

    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        image_proj,
        audio_proj,
    )

    m,u = net.load_state_dict(
        torch.load(
            os.path.join(audio_ckpt_dir, "net.pth"),
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("loaded weight from ", os.path.join(audio_ckpt_dir, "net.pth"))

    # 5. inference
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=val_noise_scheduler,
        image_proj=net.imageproj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    audio_emb = process_audio_emb(audio_emb)
    print('inference 356 audio_emb.shape:', audio_emb.shape, flush=True) # 192, 5, 12, 768

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


    times = audio_emb.shape[0] // clip_length
    # print('inference 378 times:', times, flush=True) # 12
    # assert 0

    tensor_result = []

    audio_similarities = []


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
        print('inference 409 audio_tensor.shape:', audio_tensor.shape, flush=True) # 16, 5, 12, 768
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(
            device=net.audioproj.device, dtype=net.audioproj.dtype)
        audio_tensor = net.audioproj(audio_tensor)
        print('inference 414 audio_tensor.shape:', audio_tensor.shape, flush=True) # 1, 16, 32, 768
        
        # Compute similarity matrix between frames
        # similarity_matrix = compute_frame_similarity(audio_tensor)
        # print(f"Similarity matrix shape: {similarity_matrix.shape}")  # Should be [16, 16]
        # audio_similarities.append(similarity_matrix.cpu().numpy())
        # continue

        # Optionally visualize or save the similarity matrix
        # You could add code here to visualize the matrix using matplotlib
        # or save it for later analysis
        
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
            ############ ours ############
            mask_fg=source_fg_mask,
            bg_cache_interval=config.bg_cache_kwargs.bg_cache_interval,
            use_masked_attention=config.bg_cache_kwargs.use_masked_attention,
            masked_layers=config.bg_cache_kwargs.masked_layers,
            remove_ref_list=config.bg_cache_kwargs.remove_ref_list,
            ############ parallel unet ############
            estimate_input_latents=config.parallel_unet_kwargs.estimate_input_latents,
            ############ save params ############
            save_dir=config.save_kwargs.save_dir,
            save_layers=config.save_kwargs.save_layers,
            save_types=config.save_kwargs.save_types,
            # parallel unet
            max_parallel_per_gpu=config.parallel_unet_kwargs.max_parallel_per_gpu,
        )
        
        tensor_result.append(pipeline_output.videos)
        print('inference 430 pipeline_output.videos.shape:', pipeline_output.videos.shape, flush=True) # 1, 16, 256, 256
        # np.save('tensor_result_parallel_single_2.npy', pipeline_output.videos.detach().cpu().numpy())
        # assert 0

        # Save the current pipeline output videos
        save_dir = "intermediate"
        os.makedirs(save_dir, exist_ok=True)
        tensor_to_video(pipeline_output.videos.squeeze(0), os.path.join(save_dir, f"output_{t}.mp4"), driving_audio_path)
        assert 0

    # # Save audio similarities
    # np.save('audio_similarities.npy', audio_similarities)
    # assert 0

    tensor_result = torch.cat(tensor_result, dim=2)
    tensor_result = tensor_result.squeeze(0)

    print('inference 470 tensor_result.shape:', tensor_result.shape, flush=True) 
    tensor_result = tensor_result[:, :audio_length]
    print('inference 472 tensor_result.shape:', tensor_result.shape, flush=True) 

    # np.save('tensor_result.npy', tensor_result.cpu().numpy())

    output_file = config.output
    # save the result after all iteration
    tensor_to_video(tensor_result, output_file, driving_audio_path)
    return output_file


def compute_frame_similarity(audio_tensor):
    """
    Compute cosine similarity between frames in the audio tensor.
    
    Parameters:
        audio_tensor (torch.Tensor): Tensor of shape [batch_size, video_length, feature_dim_1, feature_dim_2]
    
    Returns:
        torch.Tensor: Similarity matrix of shape [video_length, video_length]
    """
    # Extract the first batch
    features = audio_tensor[0]  # Shape: [video_length, feature_dim_1, feature_dim_2]
    
    # Flatten the features for each frame
    video_length = features.shape[0]
    flattened_features = features.reshape(video_length, -1)  # Shape: [video_length, feature_dim_1 * feature_dim_2]
    
    # Normalize the features
    normalized_features = torch.nn.functional.normalize(flattened_features, p=2, dim=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    
    return similarity_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", default="configs/inference/default.yaml")
    parser.add_argument("--source_image", type=str, required=False,
                        help="source image")
    parser.add_argument("--driving_audio", type=str, required=False,
                        help="driving audio")
    parser.add_argument(
        "--output", type=str, help="output video file name", default=".cache/output.mp4")
    parser.add_argument(
        "--pose_weight", type=float, help="weight of pose", required=False)
    parser.add_argument(
        "--face_weight", type=float, help="weight of face", required=False)
    parser.add_argument(
        "--lip_weight", type=float, help="weight of lip", required=False)
    parser.add_argument(
        "--face_expand_ratio", type=float, help="face region", required=False)
    parser.add_argument(
        "--audio_ckpt_dir", "--checkpoint", type=str, help="specific checkpoint dir", required=False)


    command_line_args = parser.parse_args()

    inference_process(command_line_args)