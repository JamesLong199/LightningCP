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

from contextlib import nullcontext
from torch.utils.flop_counter import FlopCounterMode

def register_time(model, time):
    """
    Register current timestep in model.
    
    Args:
        model: UNet model
        time: Current timestep
    """
    setattr(model, 'order', time)

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


    def process_on_gpu(self, 
                       gpu_id, 
                       latent_model_input, 
                       t_batch, 
                       encoder_hidden_states, 
                       cond, # faster diffusion
                       face_mask, 
                       pixel_values_full_mask, 
                       pixel_values_face_mask,
                       pixel_values_lip_mask, 
                       audio_tensor, 
                       motion_scale,
                       result_dict, 
                       batch_idx,
                       log_file,
                       log_flops,
                       mode):
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
        }

        # Move all inputs to the device at once
        if gpu_id != 0:
            inputs = move_tensors_to_device(inputs, device)
        
        # Process on this GPU
        with torch.no_grad(), thread_flops_counter:
            noise_pred = denoising_unet(
                sample=inputs['latent_model_input'],
                timestep=inputs['t_batch'],
                encoder_hidden_states=inputs['encoder_hidden_states'],
                cond=cond, # faster diffusion
                mask_cond_fea=inputs['mask_cond_fea'],
                full_mask=inputs['full_mask'],
                face_mask=inputs['face_mask'],
                lip_mask=inputs['lip_mask'],
                audio_embedding=inputs['audio_embedding'],
                motion_scale=inputs['motion_scale'],
                return_dict=False,
                mode=mode, # multi-gpu
            )[0]

        # Calculate FLOPs for this thread
        thread_flops = 0
        if log_flops:
            thread_flops = thread_flops_counter.get_total_flops() / 1e9

        # Move results back to CPU to combine later
        result_dict[batch_idx] = {
            'noise_pred': noise_pred.to(self.denoising_unet.device),
            'flops': thread_flops,
        }


 
    @torch.no_grad()
    def call(
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
        num_images_per_prompt=1,
        eta: float = 0.0,
        motion_scale: Optional[List[torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        estimate_input_latents: bool = False,
        prior_noise_injection: bool = False,
        cond: List[int] = [],
        # max_parallel_per_gpu: int = 3,
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
        clip_image_embeds = clip_image_embeds.to(self.image_proj.device, self.image_proj.dtype)

        encoder_hidden_states = self.image_proj(clip_image_embeds)
        uncond_encoder_hidden_states = self.image_proj(torch.zeros_like(clip_image_embeds))

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

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

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = rearrange(ref_image, "b f c h w -> (b f) c h w")
        ref_image_tensor = self.ref_image_processor.preprocess(ref_image_tensor, height=height, width=width)
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        face_mask = face_mask.unsqueeze(1).to(dtype=self.face_locator.dtype, device=self.face_locator.device)
        face_mask = repeat(face_mask, "b f c h w -> b (repeat f) c h w", repeat=video_length)
        face_mask = face_mask.transpose(1, 2)  # (bs, c, f, H, W)
        face_mask = self.face_locator(face_mask)
        face_mask = torch.cat([torch.zeros_like(face_mask), face_mask], dim=0) if do_classifier_free_guidance else face_mask

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
        
        # Convert masks to the right device and dtype
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

        uncond_audio_tensor = torch.zeros_like(audio_tensor)
        audio_tensor = torch.cat([uncond_audio_tensor, audio_tensor], dim=0)
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

        # Initialize variables for processing
        curr_step = 0
        all_steps = len(timesteps)

        self.reference_unet(
            ref_image_latents.repeat(
                (2 if do_classifier_free_guidance else 1), 1, 1, 1
            ),
            torch.zeros_like(timesteps[0]),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer)

        if self.multi_gpu_denoising_unets is not None:
            print('self.multi_gpu_denoising_unets.keys():', self.multi_gpu_denoising_unets.keys(), flush=True)
            for gpu_id, gpu_control in reference_controls.items():
                print('gpu_id:', gpu_id, flush=True)
                gpu_control.update(reference_control_writer, device=self.multi_gpu_denoising_unets[gpu_id].device)
        
        # Main denoising loop with accelerated inference
        init_latents = latents.detach().clone()
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        num_gpus = len(self.gpu_ids)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            while curr_step < all_steps:

                register_time(self.denoising_unet, curr_step)

                print('curr_step:', curr_step, flush=True)
                print('self.denoising_unet.order:', self.denoising_unet.order, flush=True)

                if self.multi_gpu_denoising_unets is not None:
                    for idx in self.multi_gpu_denoising_unets.keys():
                        register_time(self.multi_gpu_denoising_unets[idx], curr_step)
                        print(f'self.multi_gpu_denoising_unets[{idx}].order:', self.multi_gpu_denoising_unets[idx].order, flush=True)

                time_ls = [timesteps[curr_step]]
                # print('time_ls:', time_ls, flush=True)
                curr_step += 1
                
                # Collect timesteps until next key step
                while curr_step not in cond:
                    if curr_step<all_steps and len(time_ls) < self.max_parallel_per_gpu*num_gpus:
                        time_ls.append(timesteps[curr_step])
                        curr_step += 1
                    else:
                        break

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                print('len(time_ls):', len(time_ls), flush=True)
                print('time_ls:', time_ls, flush=True)

                if log_latency:
                    start_event.record()

            
                if len(time_ls) == 1 or self.multi_gpu_denoising_unets is None:
                    # Predict noise for all timesteps in parallel
                    with flops_counter:
                        noise_pred = self.denoising_unet(
                            sample=latent_model_input,
                            timestep=time_ls,
                            encoder_hidden_states=encoder_hidden_states,
                            cond=cond,
                            mask_cond_fea=face_mask,
                            full_mask=pixel_values_full_mask,
                            face_mask=pixel_values_face_mask,
                            lip_mask=pixel_values_lip_mask,
                            audio_embedding=audio_tensor,
                            motion_scale=motion_scale,
                            return_dict=False,
                        )[0]
                    
                    if log_flops:
                        gflops = flops_counter.get_total_flops() / 1e9

                else:
                    with flops_counter:
                        self.denoising_unet(
                            sample=latent_model_input,
                            timestep=time_ls,
                            encoder_hidden_states=encoder_hidden_states,
                            cond=cond,
                            mask_cond_fea=face_mask,
                            full_mask=pixel_values_full_mask,
                            face_mask=pixel_values_face_mask,
                            lip_mask=pixel_values_lip_mask,
                            audio_embedding=audio_tensor,
                            motion_scale=motion_scale,
                            return_dict=False,
                            mode='down_only',
                        )

                    if log_flops:
                        gflops_down = flops_counter.get_total_flops() / 1e9
                        print(f"UNet encoder took {gflops_down} GFLOPs", flush=True)
                    
                    # with flops_counter:
                    for idx in self.multi_gpu_denoising_unets.keys():
                        # self.multi_gpu_denoising_unets[idx].skip_feature = self.denoising_unet.skip_feature.to(self.multi_gpu_denoising_unets[idx].device)
                        self.multi_gpu_denoising_unets[idx].skip_feature = tuple(
                            t.to(self.multi_gpu_denoising_unets[idx].device) for t in self.denoising_unet.skip_feature
                        )
                        self.multi_gpu_denoising_unets[idx].toup_feature = self.denoising_unet.toup_feature.to(self.multi_gpu_denoising_unets[idx].device)
                        
                    batches_per_gpu = [0] * num_gpus
                    for k in range(len(time_ls)):
                        batches_per_gpu[k % num_gpus] += 1     

                    # Prepare input batches for each GPU
                    gpu_batches = []
                    t_index = 0
                    for gpu_idx, batch_size in enumerate(batches_per_gpu):
                        if batch_size > 0:
                            # print('batch_size:', batch_size, flush=True)
                            # print('latent_model_input.shape:', latent_model_input.shape, flush=True)
                            # assert 0
                            t_batch = time_ls[t_index:t_index+batch_size]
                            # latent_model_input_batch = latent_model_input[t_index:t_index+batch_size*2] if do_classifier_free_guidance else latent_model_input[t_index:t_index+batch_size]
                            gpu_batches.append((gpu_idx, t_batch, latent_model_input))
                            t_index += batch_size               

                    # Process each batch on its assigned GPU
                    threads = []
                    results = {}

                    for batch_idx, (gpu_idx, t_batch, latent_model_input_batch) in enumerate(gpu_batches):    
                        thread = threading.Thread(
                            target=self.process_on_gpu,
                            kwargs={
                                'gpu_id': self.gpu_ids[gpu_idx], 
                                'latent_model_input': latent_model_input_batch,
                                't_batch': t_batch,
                                'encoder_hidden_states': encoder_hidden_states,
                                'cond': cond,
                                'face_mask': face_mask,
                                'pixel_values_full_mask': pixel_values_full_mask,
                                'pixel_values_face_mask': pixel_values_face_mask,
                                'pixel_values_lip_mask': pixel_values_lip_mask,
                                'audio_tensor': audio_tensor,
                                'motion_scale': motion_scale,
                                'result_dict': results,
                                'batch_idx': batch_idx,
                                'mode': 'up_only',
                                'log_file': log_file,
                                'log_flops': log_flops,
                            }
                        )
                        threads.append(thread)
                        thread.start()
                    
                    # Wait for all threads to complete
                    for thread in threads:
                        thread.join()
                
                    # Combine results in the correct order
                    combined_noise_pred = []
                    gflops_up = 0
                    # print('face_animate 808 results.keys():', results.keys(), flush=True)
                    # print('face_animate 809 len(gpu_batches):', len(gpu_batches), flush=True)
                    for batch_idx in range(len(gpu_batches)):
                        batch_results = results[batch_idx]
                        combined_noise_pred.append(batch_results['noise_pred'])
                        # if log_flops:
                        gflops_up += batch_results['flops']

                    
                    # Concatenate noise predictions
                    noise_pred = torch.cat(combined_noise_pred, dim=0)
                    
                    if log_flops:
                        # gflops_up = flops_counter.get_total_flops() / 1e9
                        print(f"UNet decoder took {gflops_up} GFLOPs", flush=True)
                        gflops = gflops_down + gflops_up

                if log_latency:
                    end_event.record()
                    end_event.synchronize()
                    time_elapsed = start_event.elapsed_time(end_event) / 1000
                    print(f"Denoising UNet step took {time_elapsed} seconds", flush=True)
                    denoising_times.append(time_elapsed)

                if log_flops:
                    print(f"Denoising UNet step took {gflops} GFLOPs", flush=True)
                    flops_list.append(gflops)

                # Process guidance and compute denoised samples
                if do_classifier_free_guidance:
                    if len(time_ls) > 1:
                        noise_pred_uncond = noise_pred[::2]
                        noise_pred_text = noise_pred[1::2]
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                bs = noise_pred.shape[0]
                bs_perstep = bs//len(time_ls)

                # Process each timestep sequentially to update latents
                denoised_latent = latents
                for i, timestep in enumerate(time_ls):
                    # print('timestep: ', timestep)
                    if prior_noise_injection:
                        if timestep/1000 < 0.5:
                            print(f'timestep: {timestep}: using prior noise injection')
                            # print('timestep: ', timestep)
                            denoised_latent = denoised_latent + 0.003*init_latents
                    curr_noise = noise_pred[i*bs_perstep:(i+1)*bs_perstep]
                    denoised_latent = self.scheduler.step(curr_noise, timestep, denoised_latent, **extra_step_kwargs, return_dict=False)[0]
                
                latents = denoised_latent

                progress_bar.update()
            
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
                        f.write(f'denoising_time: {denoising_time}\n')
                    # Calculate and print the average denoising time
                    avg_denoising_time = sum(denoising_times) / num_inference_steps
                    f.write(f"Average denoising time per step: {avg_denoising_time} seconds\n")
            if log_flops:
                with open(log_file, 'a') as f:
                    for i, gflops in enumerate(flops_list):
                        f.write(f'gflops: {gflops}\n')
                    # Calculate and print the average flops
                    avg_flops = sum(flops_list) / num_inference_steps
                    f.write(f"Average flops per step: {avg_flops} GFLOPs\n")
        
        return FaceAnimatePipelineOutput(videos=images)
    
    