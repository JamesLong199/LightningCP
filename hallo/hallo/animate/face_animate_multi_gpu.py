# pylint: disable=R0801
"""
This module is responsible for animating faces in videos using a combination of deep learning techniques.
It provides a pipeline for generating face animations by processing video frames and extracting face features. 
The module utilizes various schedulers and utilities for efficient face animation and supports different types 
    of latents for more control over the animation process.

Functions and Classes:
- FaceAnimatePipeline: A class that extends the DiffusionPipeline class from the diffusers library to handle face animation tasks.
  - __init__: Initializes the pipeline with the necessary components (VAE, UNets, face locator, etc.).
  - prepare_latents: Generates or loads latents for the animation process, scaling them according to the scheduler's requirements.
  - prepare_extra_step_kwargs: Prepares extra keyword arguments for the scheduler step, ensuring compatibility with different schedulers.
  - decode_latents: Decodes the latents into video frames, ready for animation.

Usage:
- Import the necessary packages and classes.
- Create a FaceAnimatePipeline instance with the required components.
- Prepare the latents for the animation process.
- Use the pipeline to generate the animated video.

Note:
- This module is designed to work with the diffusers library, which provides the underlying framework for face animation using deep learning.
- The module is intended for research and development purposes, and further optimization and customization may be required for specific use cases.
"""

import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import (DDIMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat
from tqdm import tqdm

from hallo.models.mutual_self_attention import ReferenceAttentionControl
import time
import threading
# from hallo.models.tome_patch import apply_patch, ToMeReferenceAttentionControl
# from torch.utils.flop_counter import FlopCounterMode

from contextlib import nullcontext
from torch.utils.flop_counter import FlopCounterMode

def sample_gaussian_centered(n=1000, sample_size=100, std_dev=100):
    samples = []
    
    while len(samples) < sample_size:
        # Sample from a Gaussian centered at n/2
        sample = int(np.random.normal(loc=n/2, scale=std_dev))
        
        # Check if the sample is in bounds
        if 1 <= sample < n and sample not in samples:
            samples.append(sample)
    
    return samples

def sample_from_quad(total_numbers, n_samples, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace(0, total_numbers**(1/pow), n_samples+1)

        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = np.unique(np.int32(x_values**pow))[:-1]
        if len(indices) == n_samples:
            break
        pow -=0.02
    if pow <= 1:
        raise ValueError("Cannot find suitable pow. Please adjust n_samples or decrease center.")
    return indices, pow

def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        indices = [0] + [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        pow -=0.02
    if pow <= 1:
        raise ValueError("Cannot find suitable pow. Please adjust n_samples or decrease center.")
    return indices, pow

def get_next_indices(interval_seq, i, max_parallel=float('inf')):
    # Case 1: i is in interval_seq - standard behavior with max_parallel limit
    if i in interval_seq:
        current_idx = interval_seq.index(i)
        # Check if we're at the last item
        if current_idx == len(interval_seq) - 1:
            return []
        
        next_item = interval_seq[current_idx + 1]
        # Return list truncated to max_parallel
        return list(range(i + 1, min(i + 1 + max_parallel, next_item)))
    
    # Case 2: i is not in interval_seq - find the appropriate interval
    else:
        # Find the enclosing interval
        for idx in range(len(interval_seq) - 1):
            current = interval_seq[idx]
            next_item = interval_seq[idx + 1]
            
            # Check if i falls between two interval points
            if current < i < next_item:
                # Continue from i to the next_item, limited by max_parallel
                return list(range(i, min(i + max_parallel, next_item)))
        
        # If i is after the last interval point, return empty list
        if interval_seq and i > interval_seq[-1]:
            return []
            
        # If i is before the first interval point
        if interval_seq and i < interval_seq[0]:
            return list(range(i, min(i + max_parallel, interval_seq[0])))
            
        return []
@dataclass
class FaceAnimatePipelineOutput(BaseOutput):
    """
    FaceAnimatePipelineOutput is a custom class that inherits from BaseOutput and represents the output of the FaceAnimatePipeline.
    
    Attributes:
        videos (Union[torch.Tensor, np.ndarray]): A tensor or numpy array containing the generated video frames.
    
    Methods:
        __init__(self, videos: Union[torch.Tensor, np.ndarray]): Initializes the FaceAnimatePipelineOutput object with the generated video frames.
    """
    videos: Union[torch.Tensor, np.ndarray]
    denoising_times: Optional[List[float]] = None
    # added
    # time_elapsed: Optional[float] = None
    # gflops: Optional[float] = None

class FaceAnimatePipeline(DiffusionPipeline):
    """
    FaceAnimatePipeline is a custom DiffusionPipeline for animating faces.
    
    It inherits from the DiffusionPipeline class and is used to animate faces by
    utilizing a variational autoencoder (VAE), a reference UNet, a denoising UNet,
    a face locator, and an image processor. The pipeline is responsible for generating
    and animating face latents, and decoding the latents to produce the final video output.
    
    Attributes:
        vae (VaeImageProcessor): Variational autoencoder for processing images.
        reference_unet (nn.Module): Reference UNet for mutual self-attention.
        denoising_unet (nn.Module): Denoising UNet for image denoising.
        face_locator (nn.Module): Face locator for detecting and cropping faces.
        image_proj (nn.Module): Image projector for processing images.
        scheduler (Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
                         EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                         DPMSolverMultistepScheduler]): Diffusion scheduler for
                         controlling the noise level.
    
    Methods:
        __init__(self, vae, reference_unet, denoising_unet, face_locator,
                 image_proj, scheduler): Initializes the FaceAnimatePipeline
                 with the given components and scheduler.
        prepare_latents(self, batch_size, num_channels_latents, width, height,
                       video_length, dtype, device, generator=None, latents=None):
                       Prepares the initial latents for video generation.
        prepare_extra_step_kwargs(self, generator, eta): Prepares extra keyword
                       arguments for the scheduler step.
        decode_latents(self, latents): Decodes the latents to produce the final
                       video output.
    """
    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        face_locator,
        image_proj,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        multi_gpu_denoising_unets=None,  # Add this
        gpu_ids=None,                   # Add this
        max_parallel_per_gpu=1,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            face_locator=face_locator,
            scheduler=scheduler,
            image_proj=image_proj,
        )

        # Store multi-GPU models
        self.multi_gpu_denoising_unets = multi_gpu_denoising_unets
        self.gpu_ids = gpu_ids

        self.vae_scale_factor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True,
        )

        self.max_parallel_per_gpu = max_parallel_per_gpu

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_latents(
        self,
        batch_size: int,                      # Number of videos to generate in parallel
        num_channels_latents: int,           # Number of channels in the latents
        width: int,                           # Width of the video frame
        height: int,                         # Height of the video frame
        video_length: int,                   # Length of the video in frames
        dtype: torch.dtype,                 # Data type of the latents
        device: torch.device,               # Device to store the latents on
        generator: Optional[torch.Generator] = None,  # Random number generator for reproducibility
        latents: Optional[torch.Tensor] = None  # Pre-generated latents (optional)
    ):
        """
        Prepares the initial latents for video generation.

        Args:
            batch_size (int): Number of videos to generate in parallel.
            num_channels_latents (int): Number of channels in the latents.
            width (int): Width of the video frame.
            height (int): Height of the video frame.
            video_length (int): Length of the video in frames.
            dtype (torch.dtype): Data type of the latents.
            device (torch.device): Device to store the latents on.
            generator (Optional[torch.Generator]): Random number generator for reproducibility.
            latents (Optional[torch.Tensor]): Pre-generated latents (optional).

        Returns:
            latents (torch.Tensor): Tensor of shape (batch_size, num_channels_latents, width, height)
            containing the initial latents for video generation.
        """
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        Prepares extra keyword arguments for the scheduler step.

        Args:
            generator (Optional[torch.Generator]): Random number generator for reproducibility.
            eta (float): The eta (η) parameter used with the DDIMScheduler. 
            It corresponds to η in the DDIM paper (https://arxiv.org/abs/2010.02502) and should be between [0, 1].

        Returns:
            dict: A dictionary containing the extra keyword arguments for the scheduler step.
        """
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        """
        Decode the latents to produce a video.

        Parameters:
        latents (torch.Tensor): The latents to be decoded.

        Returns:
        video (torch.Tensor): The decoded video.
        video_length (int): The length of the video in frames.
        """
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(
                latents[frame_idx: frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video


    def process_on_gpu(self, gpu_id, latent_model_input, t_batch, encoder_hidden_states, 
                    face_mask, pixel_values_full_mask, pixel_values_face_mask,
                    pixel_values_lip_mask, audio_tensor, motion_scale, prv_features,
                    mask_fg, use_masked_attention, use_cached_bg, masked_layers,
                    remove_ref_list, quick_replicate, superfast_replicate,
                    cache_layer_id, cache_block_id, bg_features,
                    save_dir, save_layers, save_types, result_dict, batch_idx,
                    log_file, log_flops):
        """Process a batch of timesteps on a specific GPU"""
        
        # Define a local recursive function to move tensors to device
        def move_tensors_to_device(data, device):
            """Recursively move all tensors to the specified device."""
            if isinstance(data, torch.Tensor):
                return data.to(device)
            elif isinstance(data, dict):
                return {k: move_tensors_to_device(v, device) for k, v in data.items()}
            elif isinstance(data, list):
                return [move_tensors_to_device(item, device) for item in data]
            elif isinstance(data, tuple):
                return tuple(move_tensors_to_device(item, device) for item in data)
            else:
                return data
        
        # Set device
        device = f"cuda:{gpu_id}"
        
        # Select the appropriate model for this GPU
        if gpu_id == 0:
            denoising_unet = self.denoising_unet
        else:
            denoising_unet = self.multi_gpu_denoising_unets[gpu_id]

        # Set up FLOP counter for this thread if needed
        if log_flops:
            thread_flops_counter = FlopCounterMode([denoising_unet], depth=None, display=False)
        else:
            thread_flops_counter = nullcontext()

        # Package all inputs in a dictionary
        inputs = {
            'latent_model_input': latent_model_input,
            't_batch': t_batch,
            'encoder_hidden_states': encoder_hidden_states,
            'mask_cond_fea': face_mask,
            'full_mask': pixel_values_full_mask,
            'face_mask': pixel_values_face_mask,
            'lip_mask': pixel_values_lip_mask,
            'audio_embedding': audio_tensor,
            'motion_scale': motion_scale,
            'replicate_prv_feature': prv_features,
            'mask_fg': mask_fg,
            'bg_features': bg_features
        }

        # Move all inputs to the device at once
        if gpu_id != 0:
            inputs = move_tensors_to_device(inputs, device)
        
        # Process on this GPU
        with torch.no_grad(), thread_flops_counter:
            noise_pred, new_prv_features = denoising_unet(
                sample=inputs['latent_model_input'],
                timestep=inputs['t_batch'],
                encoder_hidden_states=inputs['encoder_hidden_states'],
                mask_cond_fea=inputs['mask_cond_fea'],
                full_mask=inputs['full_mask'],
                face_mask=inputs['face_mask'],
                lip_mask=inputs['lip_mask'],
                audio_embedding=inputs['audio_embedding'],
                motion_scale=inputs['motion_scale'],
                return_dict=False,
                ############
                mask_fg=inputs['mask_fg'],
                use_masked_attention=use_masked_attention,
                use_cached_bg=use_cached_bg,
                masked_layers=masked_layers,
                remove_ref_list=remove_ref_list,
                ############### DeepCache ###############
                replicate_prv_feature=inputs['replicate_prv_feature'],
                quick_replicate=quick_replicate,
                superfast_replicate=superfast_replicate,
                cache_layer_id=cache_layer_id,
                cache_block_id=cache_block_id,
                ############### BG Cache ###############
                bg_features=inputs['bg_features'],
                ############### Save ###############
                save_dir=save_dir,
                save_layers=save_layers,
                save_types=save_types,
            )

        # Calculate FLOPs for this thread
        thread_flops = 0
        if log_flops:
            thread_flops = thread_flops_counter.get_total_flops() / 1e9

        # Move results back to CPU to combine later
        result_dict[batch_idx] = {
            'noise_pred': noise_pred.to(self.denoising_unet.device),
            'prv_features': new_prv_features.to(self.denoising_unet.device),
            'flops': thread_flops,
        }


    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        face_emb,
        audio_tensor,
        face_mask,
        pixel_values_full_mask,
        pixel_values_face_mask,
        pixel_values_lip_mask,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        mask_fg=None, # added: {64: mask_64, 32: mask_32}
        bg_cache_interval: int = 1,
        use_masked_attention: bool = False,
        masked_layers: List[int] = {'spatial': [], 'audio': [], 'temporal': []},
        remove_ref_list: List[int] = [],
        cache_interval: int = 1,
        cache_layer_id: int = None,
        cache_block_id: int = None,
        uniform: bool = True,
        pow: float = None,
        interval_seq: List[int] = [],
        center: int = None,
        num_images_per_prompt=1,
        eta: float = 0.0,
        motion_scale: Optional[List[torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        # save
        save_dir: Optional[str] = '',
        save_layers: Optional[List[str]] = [],
        save_types: Optional[List[str]] = [],
        # parallel unet
        # multi_gpu_denoising_unets=None,
        # gpu_ids: Optional[List[int]] = None,
        estimate_input_latents: bool = False,
        ############# test efficiency ############
        log_file=None,
        log_latency=False,
        log_flops=False,     
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # prepare clip image embeddings
        clip_image_embeds = face_emb
        clip_image_embeds = clip_image_embeds.to(self.image_proj.device, self.image_proj.dtype) # 1, 512
        # print('face_animate line 297 clip_image_embeds.shape:', clip_image_embeds.shape, flush=True)

        encoder_hidden_states = self.image_proj(clip_image_embeds) # 1, 4, 768
        uncond_encoder_hidden_states = self.image_proj(torch.zeros_like(clip_image_embeds)) # 1, 4, 768
        # print('face_animate line 302 encoder_hidden_states.shape:', encoder_hidden_states.shape, flush=True)
        # print('face_animate line 303 uncond_encoder_hidden_states.shape:', uncond_encoder_hidden_states.shape, flush=True)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0) # 2, 4, 768
            # print('face_animate line 307 encoder_hidden_states.shape (after concat):', encoder_hidden_states.shape, flush=True)

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        # apply_patch(self.denoising_unet, ratio=0.25)
        # reference_control_reader = apply_patch(self.denoising_unet, ratio=0.5)
        # reference_control_reader = ToMeReferenceAttentionControl(
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            # face_masks=face_masks,
        )

        # Now create reference controls for models on other GPUs
        # reference_controls = {0: reference_control_reader}  # Store the main one
        reference_controls = {}
        # Apply to models on other GPUs if available
        if self.multi_gpu_denoising_unets is not None:
            for gpu_id, gpu_model in self.multi_gpu_denoising_unets.items():
                gpu_control = ReferenceAttentionControl(
                    gpu_model,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    mode="read",
                    batch_size=batch_size,
                    fusion_blocks="full",
                )
                reference_controls[gpu_id] = gpu_control

        num_channels_latents = self.denoising_unet.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
        )

        if interval_seq is None:
            if cache_interval == 1:
                interval_seq = list(range(num_inference_steps))
            else:
                if uniform:
                    interval_seq = list(range(0, num_inference_steps, cache_interval))
                else:
                    num_slow_step = num_inference_steps//cache_interval
                    if num_inference_steps%cache_interval != 0:
                        num_slow_step += 1
                    
                    interval_seq, pow = sample_from_quad_center(num_inference_steps, num_slow_step, center=center, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                    #interval_seq, pow = sample_from_quad(num_inference_steps, num_inference_steps//cache_interval, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                    # assert 0

        # print('cache_interval:', cache_interval)
        interval_seq = sorted(interval_seq)
        prv_features = None

        # interval_seq = [0, 3, 6, 9, 12, 18, 24, 30, 36, 38] # non-uniform 1
        # interval_seq = [0, 3, 6, 9, 12, 15, 18, 24, 30, 36, 38] # non-uniform 2
        # interval_seq = [0, 3, 6, 9, 12, 15, 18, 21, 33, 36, 39] # non-uniform 3

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = rearrange(ref_image, "b f c h w -> (b f) c h w") # 3, 3, 512, 512
        # print('face_animate line 337 ref_image_tensor.shape:', ref_image_tensor.shape, flush=True)
        ref_image_tensor = self.ref_image_processor.preprocess(ref_image_tensor, height=height, width=width)  # (bs, c, width, height) 3, 3, 512, 512
        # print('face_animate line 339 ref_image_tensor.shape (after preprocess):', ref_image_tensor.shape, flush=True)
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean 
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w) 3, 4, 64, 64
        # print('face_animate line 343 ref_image_latents.shape:', ref_image_latents.shape, flush=True)


        face_mask = face_mask.unsqueeze(1).to(dtype=self.face_locator.dtype, device=self.face_locator.device) # (bs, f, c, H, W)
        # print('face_animate line 348 face_mask.shape:', face_mask.shape, flush=True)
        face_mask = repeat(face_mask, "b f c h w -> b (repeat f) c h w", repeat=video_length)
        # print('face_animate line 350 face_mask.shape:', face_mask.shape, flush=True)    
        face_mask = face_mask.transpose(1, 2)  # (bs, c, f, H, W)
        # print('face_animate line 352 face_mask.shape:', face_mask.shape, flush=True)
        face_mask = self.face_locator(face_mask)
        # print('face_animate line 354 face_mask.shape:', face_mask.shape, flush=True)
        face_mask = torch.cat([torch.zeros_like(face_mask), face_mask], dim=0) if do_classifier_free_guidance else face_mask # 2, 320, 16, 64, 64
        # print('face_animate line 356 face_mask.shape:', face_mask.shape, flush=True)

        pixel_values_full_mask = (
            [torch.cat([mask] * 2) for mask in pixel_values_full_mask]
            if do_classifier_free_guidance
            else pixel_values_full_mask
        )
        pixel_values_face_mask = (
            [torch.cat([mask] * 2) for mask in pixel_values_face_mask]
            if do_classifier_free_guidance
            else pixel_values_face_mask
        )
        pixel_values_lip_mask = (
            [torch.cat([mask] * 2) for mask in pixel_values_lip_mask]
            if do_classifier_free_guidance
            else pixel_values_lip_mask
        )
        pixel_values_face_mask_ = []
        for mask in pixel_values_face_mask:
            pixel_values_face_mask_.append(
                mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_face_mask = pixel_values_face_mask_
        pixel_values_lip_mask_ = []
        for mask in pixel_values_lip_mask:
            pixel_values_lip_mask_.append(
                mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_lip_mask = pixel_values_lip_mask_
        pixel_values_full_mask_ = []
        for mask in pixel_values_full_mask:
            pixel_values_full_mask_.append(
                mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_full_mask = pixel_values_full_mask_


        uncond_audio_tensor = torch.zeros_like(audio_tensor) # 1, 16, 32, 768
        # print('face_animate line 358 uncond_audio_tensor.shape:', uncond_audio_tensor.shape, flush=True)    
        audio_tensor = torch.cat([uncond_audio_tensor, audio_tensor], dim=0) # 2, 16, 32, 768
        # print('face_animate line 360 audio_tensor.shape:', audio_tensor.shape, flush=True)
        audio_tensor = audio_tensor.to(dtype=self.denoising_unet.dtype, device=self.denoising_unet.device)

        if log_latency:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        if log_flops:
            flops_counter = FlopCounterMode(self.denoising_unet, depth=None, display=False)
        else:
            flops_counter = nullcontext()

        denoising_times = []
        flops_list = []


        if bg_cache_interval > 1:
            bg_features = {}
        else:
            bg_features = None

        if bg_cache_interval == 1:
            interval_seq_bg = list(range(num_inference_steps))
        else:
            if uniform:
                interval_seq_bg = list(range(0, num_inference_steps, bg_cache_interval))
            else:
                num_slow_step = num_inference_steps//bg_cache_interval
                if num_inference_steps%bg_cache_interval != 0:
                    num_slow_step += 1
                
                interval_seq_bg, pow = sample_from_quad_center(num_inference_steps, num_slow_step, center=center, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                print(interval_seq_bg, len(interval_seq_bg), pow)

        interval_seq_bg = sorted(interval_seq_bg)

        # interval_seq_bg = interval_seq # non-uniform 3
        interval_seq_bg = [0, 3, 6, 9, 12, 15, 18, 21, 33, 36, 39] # hardcode for ablation experiment

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        skip_steps = 0
        init_latents = latents.detach().clone()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # if i >= 33:
                #     assert 0
                # print('skip_steps:', skip_steps, flush=True)
                if skip_steps > 0:
                    skip_steps -= 1
                    if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
                    # denoising_times.append(0.0)
                    # print('line 691 should continue')
                    continue
                # print('line 693')

                if i in interval_seq:
                    prv_features = None      
                    # bg_features = None
                
                # parallel_indices = []
                num_gpus = len(self.gpu_ids) if self.gpu_ids else 1
                parallel_timesteps = []
                if i-1 in interval_seq:
                    parallel_indices = get_next_indices(interval_seq, i-1, max_parallel=self.max_parallel_per_gpu*num_gpus)
                elif i not in interval_seq:
                    parallel_indices = get_next_indices(interval_seq, i, max_parallel=self.max_parallel_per_gpu*num_gpus)
                else:
                    parallel_indices = []

                parallel_timesteps = [timesteps[j] for j in parallel_indices]
                # print('parallel_timesteps:', parallel_timesteps, flush=True)
                
                parallel_t = [t]
                if len(parallel_timesteps)>1:
                    parallel_t=parallel_timesteps

                # Limit batch size to available GPUs * max_parallel
                num_timesteps = len(parallel_t)
                max_batch_size = num_gpus * self.max_parallel_per_gpu   # max_parallel=3 from your code
                if num_timesteps > max_batch_size:
                    # Truncate to max batch size
                    parallel_t = parallel_t[:max_batch_size]
                    num_timesteps = len(parallel_t)

                if i in interval_seq_bg:
                    use_cached_bg = False
                else:
                    use_cached_bg = True
       
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer, device=self.denoising_unet.device)
                    if self.multi_gpu_denoising_unets is not None:
                        # print('self.multi_gpu_denoising_unets.keys():', self.multi_gpu_denoising_unets.keys(), flush=True)
                        for gpu_id, gpu_control in reference_controls.items():
                            # print('gpu_id:', gpu_id, flush=True)
                            gpu_control.update(reference_control_writer, device=self.multi_gpu_denoising_unets[gpu_id].device)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # print('face_animate 754 latent_model_input.shape:', latent_model_input.shape, flush=True)

                if estimate_input_latents:
                    if i < 10:
                        # print('face animate 620 latent_model_input.shape:', latent_model_input.shape, flush=True)
                        latent_model_input = latent_model_input.repeat(len(parallel_t), 1, 1, 1, 1)
                        # print('face animate 623 latent_model_input.shape:', latent_model_input.shape, flush=True)
                    else:
                        # latent_input = latent_model_input
                        latent_model_input_list = [latent_model_input]
                        
                        for m in range(len(parallel_t)-1):
                            # print('face animate 628 latent_model_input.shape:', latent_model_input.shape, flush=True)
                            # print('face animate 631 noise_pred.shape:', noise_pred.shape, flush=True)
                            latent_model_input = self.scheduler.step(noise_pred[-1].unsqueeze(0), parallel_t[m], latent_model_input, **extra_step_kwargs, return_dict=False)[0]
                            latent_model_input_list.append(latent_model_input)
    
                        latent_model_input = torch.cat(latent_model_input_list, dim=0)
                        # np.save(f'latent_model_input_parallel.npy', latent_model_input.detach().cpu().numpy())
                        # assert 0
                        assert latent_model_input.shape[0] == 2*len(parallel_t) if do_classifier_free_guidance else len(parallel_t)
                else:
                    print('estimate_input_latents:', estimate_input_latents, flush=True)
                    latent_model_input = latent_model_input.repeat(len(parallel_t), 1, 1, 1, 1)

                if log_latency:
                    start_event.record()

                # if len(parallel_t) == 1:
                if len(parallel_t) == 1 or self.multi_gpu_denoising_unets is None:
                    with flops_counter:
                        noise_pred, prv_features = self.denoising_unet(
                            latent_model_input,
                            parallel_t,
                            encoder_hidden_states=encoder_hidden_states,
                            mask_cond_fea=face_mask,
                            full_mask=pixel_values_full_mask,
                            face_mask=pixel_values_face_mask,
                            lip_mask=pixel_values_lip_mask,
                            audio_embedding=audio_tensor,
                            motion_scale=motion_scale,
                            return_dict=False,
                            ############
                            mask_fg=mask_fg,
                            use_masked_attention=use_masked_attention,
                            use_cached_bg=use_cached_bg,
                            masked_layers=masked_layers,
                            remove_ref_list=remove_ref_list,
                            ############### DeepCache ###############
                            replicate_prv_feature=prv_features,
                            quick_replicate= cache_interval>1,
                            superfast_replicate=cache_block_id==-1,
                            cache_layer_id=cache_layer_id,
                            cache_block_id=cache_block_id,
                            ############### BG Cache ###############
                            bg_features=bg_features,
                            ############### Save ###############
                            save_dir=save_dir,
                            save_layers=save_layers,
                            save_types=save_types,
                        )

                    if log_flops:
                        gflops = flops_counter.get_total_flops() / 1e9
                        print(f"Denoising UNet step took {gflops} GFLOPs", flush=True)
                        flops_list.append(gflops)
                
                else:
                    
                    # Distribute timesteps across available GPUs
                    # num_gpus = len(self.gpu_ids) if self.gpu_ids else 1
                    # num_timesteps = len(parallel_t)
                    
                    # # Limit batch size to available GPUs * max_parallel
                    # max_batch_size = num_gpus * self.max_parallel_per_gpu   # max_parallel=3 from your code
                    # if num_timesteps > max_batch_size:
                    #     # Truncate to max batch size
                    #     parallel_t = parallel_t[:max_batch_size]
                    #     num_timesteps = len(parallel_t)
                    
                    # Determine how many batches per GPU
                    batches_per_gpu = [0] * num_gpus
                    for k in range(num_timesteps):
                        batches_per_gpu[k % num_gpus] += 1

                    # Prepare input batches for each GPU
                    gpu_batches = []
                    t_index = 0
                    for gpu_idx, batch_size in enumerate(batches_per_gpu):
                        if batch_size > 0:
                            # print('batch_size:', batch_size, flush=True)
                            # print('latent_model_input.shape:', latent_model_input.shape, flush=True)
                            # assert 0
                            t_batch = parallel_t[t_index:t_index+batch_size]
                            latent_model_input_batch = latent_model_input[t_index:t_index+batch_size*2] if do_classifier_free_guidance else latent_model_input[t_index:t_index+batch_size]
                            gpu_batches.append((gpu_idx, t_batch, latent_model_input_batch))
                            t_index += batch_size

                    # Process each batch on its assigned GPU
                    threads = []
                    results = {}

                    for batch_idx, (gpu_idx, t_batch, latent_model_input_batch) in enumerate(gpu_batches):    
                        thread = threading.Thread(
                            target=self.process_on_gpu,
                            args=(
                                self.gpu_ids[gpu_idx], 
                                latent_model_input_batch,
                                t_batch,
                                encoder_hidden_states,
                                face_mask,
                                pixel_values_full_mask,
                                pixel_values_face_mask,
                                pixel_values_lip_mask,
                                audio_tensor,
                                motion_scale,
                                prv_features,
                                mask_fg,
                                use_masked_attention,
                                use_cached_bg,
                                masked_layers,
                                remove_ref_list,
                                cache_interval > 1,  # quick_replicate
                                cache_block_id == -1,  # superfast_replicate
                                cache_layer_id,
                                cache_block_id,
                                bg_features,
                                save_dir,
                                save_layers,
                                save_types,
                                results,
                                batch_idx,
                                log_file,
                                log_flops,
                            )
                        )
                        threads.append(thread)
                        thread.start()
                    
                    # Wait for all threads to complete
                    for thread in threads:
                        thread.join()
                    
                    # Combine results in the correct order
                    combined_noise_pred = []
                    
                    gflops = 0
                    for batch_idx in range(len(gpu_batches)):
                        batch_results = results[batch_idx]
                        combined_noise_pred.append(batch_results['noise_pred'])
                        gflops += batch_results['flops']
                    
                    # Concatenate noise predictions
                    noise_pred = torch.cat(combined_noise_pred, dim=0)
                    
                    # Use the last batch's prv_features (typically from GPU 0)
                    prv_features = results[len(gpu_batches)-1]['prv_features']  

                    if log_flops:
                        print(f"Denoising UNet step took {gflops} GFLOPs", flush=True)
                        flops_list.append(gflops)

                if log_latency:
                    end_event.record()
                    end_event.synchronize()
                    time_elapsed = start_event.elapsed_time(end_event) / 1000
                    print(f"Denoising UNet step took {time_elapsed} seconds", flush=True)
                    denoising_times.append(time_elapsed)

                # perform guidance
                if do_classifier_free_guidance:
                    if len(parallel_t) > 1:
                        noise_pred_uncond = noise_pred[::2]
                        noise_pred_text = noise_pred[1::2]
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # print(f"latents.shape: {latents.shape}")
                # print(f"noise_pred.shape: {noise_pred.shape}")
                # compute the previous noisy sample x_t -> x_t-1
                # for j in range(len(parallel_t)):
                #     print('face_animate 960 noise_pred[j].device:', noise_pred[j].device, flush=True)
                for j, timestep in enumerate(parallel_t):
                    # print('face_animate 959 timestep.device:', timestep.device, flush=True)
                    # print('face_animate 960 noise_pred[j].device:', noise_pred[j].device, flush=True)
                    # print('face_animate 961 latents.device:', latents.device, flush=True)
                    # if timestep/1000 < 0.5:
                    #     latents = latents + 0.003*init_latents
                    latents = self.scheduler.step(noise_pred[j].unsqueeze(0), timestep, latents, **extra_step_kwargs, return_dict=False)[0]
                
                if len(parallel_t) > 1:
                    skip_steps = len(parallel_t) - 1

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()
        

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        if log_file is not None:
            assert not (log_latency and log_flops)
            if log_latency:
                with open(log_file, 'a') as f:
                    for i, denoising_time in enumerate(denoising_times):
                        f.write(f'timestep {i}: {denoising_time}\n')
                    # Calculate and print the average denoising time
                    avg_denoising_time = sum(denoising_times) / num_inference_steps
                    f.write(f"Average denoising time per step: {avg_denoising_time} seconds\n")
            if log_flops:
                with open(log_file, 'a') as f:
                    for i, gflops in enumerate(flops_list):
                        f.write(f'timestep {i}: {gflops}\n')
                    # Calculate and print the average flops
                    avg_flops = sum(flops_list) / num_inference_steps
                    f.write(f"Average flops per step: {avg_flops} GFLOPs\n")

        # with open(f'denoising_times.txt', 'a') as f:
        #     for i, denoising_time in enumerate(denoising_times):
        #         f.write(f'timestep {i}: {denoising_time}\n')
        #     # Calculate and print the average denoising time
        #     avg_denoising_time = sum(denoising_times) / len(num_inference_steps)
        #     f.write(f"Average denoising time per step: {avg_denoising_time} seconds\n")

        # return FaceAnimatePipelineOutput(videos=images, denoising_times=denoising_times)
        return FaceAnimatePipelineOutput(videos=images)




