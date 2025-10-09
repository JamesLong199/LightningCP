#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EchoMimic
@File    ：pipeline_echo_mimic.py
@Author  ：juzhen.czy
@Date    ：2024/3/4 17:44 
'''
# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method

from src.utils.step_func import origin_by_velocity_and_sample, psuedo_velocity_wrt_noisy_and_timestep, get_alpha # acc

from contextlib import nullcontext
from torch.utils.flop_counter import FlopCounterMode

import time
import threading

def register_time(model, time):
    """
    Register current timestep in model.
    
    Args:
        model: UNet model
        time: Current timestep
    """
    setattr(model, 'order', time)

@dataclass
class Audio2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    denoising_times: Optional[List[float]] = None


class Audio2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        audio_guider,
        face_locator,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
        # multi-gpu
        multi_gpu_denoising_unets=None,  # Add this
        gpu_ids=None,                   # Add this
        max_parallel_per_gpu=1,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            audio_guider=audio_guider,
            face_locator=face_locator,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

        # Store multi-GPU models
        self.multi_gpu_denoising_unets = multi_gpu_denoising_unets
        self.gpu_ids = gpu_ids
        self.max_parallel_per_gpu = max_parallel_per_gpu

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

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

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
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

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator
    ):
        shape = (
            batch_size,
            num_channels_latents,
            # context_frame_length,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents_seg = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )
        latents = latents_seg
        latents = torch.clamp(latents, -1.5, 1.5)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    def process_on_gpu(self, 
                       gpu_id, 
                       latent_model_input, 
                       t_batch, 
                       encoder_hidden_states, 
                       audio_cond_fea, 
                       face_musk_fea, 
                       result_dict, 
                       batch_idx,
                       cond, # faster diffusion
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
            'audio_cond_fea': audio_cond_fea,
            'face_musk_fea': face_musk_fea,
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
                audio_cond_fea=inputs['audio_cond_fea'],
                face_musk_fea=inputs['face_musk_fea'],
                return_dict=False,
                cond=cond, # faster diffusion
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
    def __call__(
        self,
        ref_image,
        audio_path,
        face_mask_tensor,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=0,
        context_batch_size=1,
        interpolation_factor=1,
        audio_sample_rate=16000,
        fps=25,
        audio_margin=2,
        acc: bool = False,
        prior_noise_injection: bool = False,
        cond: List[int] = [],
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

        # print('do_classifier_free_guidance:', do_classifier_free_guidance, flush=True)
        # print('guidance_scale:', guidance_scale, flush=True)
        # print('num_inference_steps:', num_inference_steps, flush=True)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

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

        whisper_feature = self.audio_guider.audio2feat(audio_path)

        whisper_chunks = self.audio_guider.feature2chunks(feature_array=whisper_feature, fps=fps)

        print("whisper_chunks:", whisper_chunks.shape)
        audio_frame_num = whisper_chunks.shape[0]
        audio_fea_final = torch.Tensor(whisper_chunks).to(dtype=self.vae.dtype, device=self.vae.device)
        audio_fea_final = audio_fea_final.unsqueeze(0)
        print("audio_fea_final:", audio_fea_final.shape)
        video_length = min(video_length, audio_frame_num)
        if video_length < audio_frame_num:
            audio_fea_final = audio_fea_final[:, :video_length, :, :]

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            audio_fea_final.dtype,
            device,
            generator
        )
        # print(video_length, latents.shape)
        c_face_locator_tensor = self.face_locator(face_mask_tensor)
        uc_face_locator_tensor = torch.zeros_like(c_face_locator_tensor)
        face_locator_tensor = torch.cat([uc_face_locator_tensor, c_face_locator_tensor], dim=0)
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b , 4, h, w)

        context_scheduler = get_context_scheduler(context_schedule)


        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        context_queue = list(
            context_scheduler(
                0,
                num_inference_steps,
                latents.shape[2],
                context_frames,
                context_stride,
                context_overlap,
            )
        )
        # print("ref_image_latents shape:", ref_image_latents.shape)
        # print("face_mask_tensor shape:", face_mask_tensor.shape)
        # print("face_locator_tensor shape:", face_locator_tensor.shape)

        if log_latency:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        if log_flops:
            flops_counter = FlopCounterMode(self.denoising_unet, depth=None, display=False)
        else:
            flops_counter = nullcontext()

        denoising_times = []
        flops_list = []

        # print('self.denoising_unet.reference_unet.:', self.denoising_unet.reference_unet)
        # 1. Forward reference image
        self.reference_unet(
            ref_image_latents,
            torch.zeros_like(timesteps[0]),
            encoder_hidden_states=None,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer, do_classifier_free_guidance=do_classifier_free_guidance, device=self.denoising_unet.device)

        if self.multi_gpu_denoising_unets is not None:
            print('self.multi_gpu_denoising_unets.keys():', self.multi_gpu_denoising_unets.keys(), flush=True)
            for gpu_id, gpu_control in reference_controls.items():
                print('gpu_id:', gpu_id, flush=True)
                gpu_control.update(reference_control_writer, do_classifier_free_guidance=do_classifier_free_guidance, device=self.multi_gpu_denoising_unets[gpu_id].device)
        num_gpus = len(self.gpu_ids)

        # assert 0

        # Initialize variables for processing
        curr_step = 0
        all_steps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            while curr_step < all_steps:
                toup_features = torch.zeros((2 if do_classifier_free_guidance else 1, 1280, latents.shape[2], 8, 8), device=latents.device, dtype=latents.dtype)

                # Initialize placeholder for down_block_res_samples
                down_block_res_samples = [
                    torch.zeros((2 if do_classifier_free_guidance else 1, 320, latents.shape[2], 64, 64), device=latents.device, dtype=latents.dtype),   # 0-2
                    torch.zeros((2 if do_classifier_free_guidance else 1, 320, latents.shape[2], 64, 64), device=latents.device, dtype=latents.dtype),
                    torch.zeros((2 if do_classifier_free_guidance else 1, 320, latents.shape[2], 64, 64), device=latents.device, dtype=latents.dtype),
                    torch.zeros((2 if do_classifier_free_guidance else 1, 320, latents.shape[2], 32, 32), device=latents.device, dtype=latents.dtype),   # 3
                    torch.zeros((2 if do_classifier_free_guidance else 1, 640, latents.shape[2], 32, 32), device=latents.device, dtype=latents.dtype),   # 4-5
                    torch.zeros((2 if do_classifier_free_guidance else 1, 640, latents.shape[2], 32, 32), device=latents.device, dtype=latents.dtype),
                    torch.zeros((2 if do_classifier_free_guidance else 1, 640, latents.shape[2], 16, 16), device=latents.device, dtype=latents.dtype),   # 6
                    torch.zeros((2 if do_classifier_free_guidance else 1, 1280, latents.shape[2], 16, 16), device=latents.device, dtype=latents.dtype),  # 7-8
                    torch.zeros((2 if do_classifier_free_guidance else 1, 1280, latents.shape[2], 16, 16), device=latents.device, dtype=latents.dtype),
                    torch.zeros((2 if do_classifier_free_guidance else 1, 1280, latents.shape[2], 8, 8), device=latents.device, dtype=latents.dtype),    # 9-11
                    torch.zeros((2 if do_classifier_free_guidance else 1, 1280, latents.shape[2], 8, 8), device=latents.device, dtype=latents.dtype),
                    torch.zeros((2 if do_classifier_free_guidance else 1, 1280, latents.shape[2], 8, 8), device=latents.device, dtype=latents.dtype),
                ]

                # if curr_step>0:
                #     assert 0

                register_time(self.denoising_unet, curr_step)

                # print('curr_step:', curr_step, flush=True)
                # print('self.denoising_unet.order:', self.denoising_unet.order, flush=True)

                if self.multi_gpu_denoising_unets is not None:
                    for idx in self.multi_gpu_denoising_unets.keys():
                        register_time(self.multi_gpu_denoising_unets[idx], curr_step)
                        print(f'self.multi_gpu_denoising_unets[{idx}].order:', self.multi_gpu_denoising_unets[idx].order, flush=True)
                
                time_ls = [timesteps[curr_step]]
                curr_step += 1

                # Collect timesteps until next key step
                while curr_step not in cond:
                    if curr_step<all_steps and len(time_ls) < self.max_parallel_per_gpu*num_gpus:
                        time_ls.append(timesteps[curr_step])
                        curr_step += 1
                    else:
                        break
                
                print("time_ls:", time_ls, flush=True)
                print("len(time_ls):", len(time_ls), flush=True)
                print("curr_step:", curr_step, flush=True)

                if len(time_ls) > 1:
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0]*len(time_ls) * (2 if do_classifier_free_guidance else 1),
                            *latents.shape[1:],
                        ),
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                else:
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                            *latents.shape[1:],
                        ),
                        device=latents.device,
                        dtype=latents.dtype,
                    )

                print("pipeline 658 noise_pred.shape:", noise_pred.shape, flush=True)

                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                global_context = []
                for j in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            j * context_batch_size : (j + 1) * context_batch_size
                        ]
                    )

                for context in global_context:
                    new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
                    for c_j in range(len(context)):
                        for c_i in range(len(context[c_j])):
                            new_context[c_j][c_i] = (context[c_j][c_i] + curr_step * 2) % video_length

                    # print("pipeline 682 new_context:", new_context, flush=True)
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in new_context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    # print("pipeline 688 latent_model_input.shape:", latent_model_input.shape, flush=True)

                    c_audio_latents = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)
                    audio_latents = torch.cat([torch.zeros_like(c_audio_latents), c_audio_latents], 0)

                    # latent_model_input = self.scheduler.scale_model_input(
                    #     latent_model_input, t
                    # )

                    if log_latency:
                        start_event.record()

                    
                    
                    if len(time_ls) == 1:
                        with flops_counter:
                            pred = self.denoising_unet(
                                sample=latent_model_input,
                                timestep=time_ls,
                                encoder_hidden_states=None,
                                audio_cond_fea=audio_latents if do_classifier_free_guidance else c_audio_latents,
                                face_musk_fea=face_locator_tensor if do_classifier_free_guidance else c_face_locator_tensor,
                                return_dict=False,
                                cond=cond,
                            )[0]
                            # reference_control_reader.update(reference_control_writer)

                            # print("pipeline 734 toup_features.shape:", toup_features.shape, flush=True)
                            # print("pipeline 735 self.denoising_unet.toup_feature.shape:", self.denoising_unet.toup_feature.shape, flush=True)
                            # print("pipeline 736 new_context:", new_context, flush=True)
                            for c in new_context:
                                # print('pipeline 738 c:', c, flush=True)
                                toup_features[:, :, c] = self.denoising_unet.toup_feature
                                for i in range(len(down_block_res_samples)):
                                    down_block_res_samples[i][:, :, c] = self.denoising_unet.skip_feature[i]

                        if log_flops:
                            gflops = flops_counter.get_total_flops() / 1e9
                            print(f"UNet encoder took {gflops} GFLOPs", flush=True)

                    else:
                        with flops_counter:
                            self.denoising_unet(
                                sample=latent_model_input,
                                timestep=time_ls,
                                encoder_hidden_states=None,
                                cond=cond,
                                audio_cond_fea=audio_latents if do_classifier_free_guidance else c_audio_latents,
                                face_musk_fea=face_locator_tensor if do_classifier_free_guidance else c_face_locator_tensor,
                                return_dict=False,
                                mode='down_only',
                            )

                        if log_flops:
                            gflops_down = flops_counter.get_total_flops() / 1e9
                            print(f"UNet encoder took {gflops_down} GFLOPs", flush=True)
                    

                        # probably need to store the feature of all context
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
                        # print('pipeline 743 batches_per_gpu:', batches_per_gpu, flush=True)
                        for gpu_idx, batch_size in enumerate(batches_per_gpu):
                            # print('pipeline 745 gpu_idx:', gpu_idx, 'batch_size:', batch_size, flush=True)
                            if batch_size > 0:
                                # print('batch_size:', batch_size, flush=True)
                                # print('latent_model_input.shape:', latent_model_input.shape, flush=True)
                                # assert 0
                                t_batch = time_ls[t_index:t_index+batch_size]
                                # latent_model_input_batch = latent_model_input[t_index:t_index+batch_size*2] if do_classifier_free_guidance else latent_model_input[t_index:t_index+batch_size]
                                gpu_batches.append((gpu_idx, t_batch, latent_model_input))
                                t_index += batch_size 
                        # assert 0

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
                                    'encoder_hidden_states': None,
                                    'cond': cond,
                                    'audio_cond_fea': audio_latents if do_classifier_free_guidance else c_audio_latents,
                                    'face_musk_fea': face_locator_tensor if do_classifier_free_guidance else c_face_locator_tensor,
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
                        combined_pred = []
                        gflops_up = 0
                        # print('face_animate 808 results.keys():', results.keys(), flush=True)
                        # print('face_animate 809 len(gpu_batches):', len(gpu_batches), flush=True)
                        for batch_idx in range(len(gpu_batches)):
                            batch_results = results[batch_idx]
                            combined_pred.append(batch_results['noise_pred'])
                            gflops_up += batch_results['flops']
                        
                        # Concatenate noise predictions
                        pred = torch.cat(combined_pred, dim=0)

                        if log_flops:
                            # gflops_up = flops_counter.get_total_flops() / 1e9
                            print(f"UNet decoder took {gflops_up} GFLOPs", flush=True)
                            gflops = gflops_down + gflops_up
                    
                    if log_latency:
                        end_event.record()
                        end_event.synchronize()
                        time_elapsed = start_event.elapsed_time(end_event) / 1000
                        print(f"Denoising UNet step took {time_elapsed} seconds for current context", flush=True)
                        denoising_times.append(time_elapsed)

                    if log_flops:
                        print(f"Denoising UNet step took {gflops} GFLOPs for current context", flush=True)
                        flops_list.append(gflops)

                    if acc:
                        # assert 0
                        # acc
                        alphas_cumprod = self.scheduler.alphas_cumprod.to(latent_model_input.device)
                        if len(time_ls) > 1:
                            new_pred_list = []
                            for p_t in range(len(time_ls)):
                                step_size = 2 if do_classifier_free_guidance else 1
                                x_pred = origin_by_velocity_and_sample(pred[p_t:p_t+step_size], latent_model_input[p_t:p_t+step_size], alphas_cumprod, time_ls[p_t])
                                new_pred = psuedo_velocity_wrt_noisy_and_timestep(latent_model_input[p_t:p_t+step_size], x_pred, alphas_cumprod, time_ls[p_t], torch.ones_like(time_ls[p_t]) * (-1))
                                new_pred_list.append(new_pred)                                
                            pred = torch.cat(new_pred_list, dim=0)
                        else:
                            x_pred = origin_by_velocity_and_sample(pred, latent_model_input, alphas_cumprod, time_ls[0])
                            pred = psuedo_velocity_wrt_noisy_and_timestep(latent_model_input, x_pred, alphas_cumprod, time_ls[0], torch.ones_like(time_ls[0]) * (-1))

                    for j, c in enumerate(new_context):

                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred

                        counter[:, :, c] = counter[:, :, c] + 1

                print("pipeline 823 noise_pred.shape:", noise_pred.shape, flush=True)

                # import os
                # os.makedirs('encoder_data/t{:02d}'.format(curr_step), exist_ok=True)
                # np.save('encoder_data/t{:02d}/t{:02d}_toup_features.npy'.format(curr_step, curr_step), toup_features[:,:,:12].cpu().numpy())
                # for i in range(len(down_block_res_samples)):
                #     np.save('encoder_data/t{:02d}/t{:02d}_down_block_res_samples_{}.npy'.format(curr_step, curr_step, i), down_block_res_samples[i][:,:,:12].cpu().numpy())

                # perform guidance
                if do_classifier_free_guidance:
                    if len(time_ls) > 1:
                        noise_pred_uncond = noise_pred[::2]
                        noise_pred_text = noise_pred[1::2]
                    else:
                        # print('len(time_ls) == 1', flush=True)
                        # print('time_ls:', time_ls, flush=True)
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    noise_pred = noise_pred / counter
                else:
                    noise_pred = noise_pred / counter # acc

                for j, timestep in enumerate(time_ls):
                    latents = self.scheduler.step(
                        noise_pred[j].unsqueeze(0), timestep, latents, **extra_step_kwargs
                    ).prev_sample

                if curr_step == len(timesteps) - 1 or (
                    (curr_step + 1) > num_warmup_steps and (curr_step + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                

            reference_control_reader.clear()
            reference_control_writer.clear()

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)
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
                    avg_denoising_time = sum(denoising_times) / num_inference_steps / len(global_context)
                    f.write(f"Average denoising time per step per context: {avg_denoising_time} seconds\n")
            if log_flops:
                with open(log_file, 'a') as f:
                    for i, gflops in enumerate(flops_list):
                        f.write(f'gflops: {gflops}\n')

                    # Calculate and print the average flops
                    avg_flops = sum(flops_list) / num_inference_steps / len(global_context)
                    f.write(f"Average flops per step per context: {avg_flops} GFLOPs\n")

        return Audio2VideoPipelineOutput(videos=images, denoising_times=denoising_times)

    def smooth_f_axis(self, tensor, smoothing_coef=0.2):
        """
        对5维tensor的F轴进行平滑，首尾帧保持原始值，其他帧应用平滑操作。

        参数:
        tensor -- 输入的5D Tensor，形状为(B, C, F, W, H)
        smoothing_coef -- 平滑前后帧的系数，默认为0.3

        返回:
        smoothed_tensor -- 平滑后的5D Tensor
        """
        # 生成平滑核
        weight_kernel = torch.tensor([smoothing_coef, 1 - 2 * smoothing_coef, smoothing_coef],
                                     device=tensor.device, dtype=tensor.dtype).view(1, 1, -1)

        b, c, f, w, h = tensor.size()

        # 对于第三轴（F轴）仅平滑内部帧 (1:-1)
        # 首先，移动要平滑的轴到最后一个维度以便应用conv1d
        tensor_moved = rearrange(tensor, "b c f h w -> b (c h w) f")

        # 对除了两端帧外的所有帧应用conv1d
        internal_frames = F.conv1d(tensor_moved, weight_kernel)
        print("tensor:", tensor.shape)

        # 重新整理输出形状为 (B, F, C, W, H)
        internal_frames = rearrange(internal_frames, "b (c h w) f -> b c f h w", c=c, h=h, w=w)
        print("internal_frames:", internal_frames.shape)
        # 将第一帧和最后一帧保持不变，合并结果
        # 首帧 tensor[:, :, 0:1, :, :], 中间帧 internal_frames[:, :, 1:-1, :, :], 最后帧 tensor[:, :, -1:, :, :]
        smoothed_tensor = torch.cat(
            [tensor[:, :, 0:1, :, :], internal_frames, tensor[:, :, -1:, :, :]], dim=2)

        return smoothed_tensor