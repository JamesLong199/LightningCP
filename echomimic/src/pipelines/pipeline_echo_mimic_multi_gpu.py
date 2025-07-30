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
from typing import Callable, List, Optional, Union, Dict

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

# def get_bg_features_context(c, do_classifier_free_guidance, total_length):
#     if not do_classifier_free_guidance:
#         return c
#     else:
#         if total_length-1 in c and 0 in c:
#             end_idx = c.index(total_length-1)
#             # print('end_idx:', end_idx, flush=True)
#             if end_idx == 0:
#                 new_c = [c[0]*2, c[0]*2+1] + list(range(2 * c[1], 2 * c[-1] + 2))
#             else:
#                 new_c = list(range(2 * c[0], 2 * c[end_idx] + 2)) + list(range(2 * c[end_idx+1], 2 * c[-1] + 2))
#             return new_c
#         else:
#             return list(range(2 * c[0], 2 * c[-1] + 2))
        

def get_next_indices(interval_seq, i, max_parallel=float('inf'), num_total=-1):
    # Case 1: i is in interval_seq - standard behavior with max_parallel limit
    # print('pipeline 43 interval_seq:', interval_seq, flush=True)
    # print('pipeline 44 i:', i, flush=True)

    if i in interval_seq:
        # print('pipeline 700 i in interval_seq:', i, flush=True)
        current_idx = interval_seq.index(i)
        # print('pipeline 702 current_idx:', current_idx, flush=True)
        # Check if we're at the last item
        if current_idx == len(interval_seq) - 1:
            # print('pipeline 704 current_idx == len(interval_seq) - 1:', current_idx, flush=True)
            return []

        # # If i is after the last interval point, return empty list
        # if interval_seq and i > interval_seq[-1]:
        #     assert num_total != -1
        #     print('pipeline 58 i is after the last interval point:', i, flush=True)
        #     return list(range(i, min(i + max_parallel, num_total)))
        
        next_item = interval_seq[current_idx + 1]
        # print('pipeline 706 next_item:', next_item, flush=True) 
        # print('pipeline 707 list(range(i + 1, min(i + 1 + max_parallel, next_item))):', list(range(i + 1, min(i + 1 + max_parallel, next_item))), flush=True)
        # Return list truncated to max_parallel
        return list(range(i + 1, min(i + 1 + max_parallel, next_item)))
    
    # Case 2: i is not in interval_seq - find the appropriate interval
    else:
        # print('pipeline 69 i is not in interval_seq:', i, flush=True)
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
            assert num_total != -1
            # print('pipeline 83 i is after the last interval point:', i, flush=True)
            return list(range(i, min(i + max_parallel, num_total)))
            
        # If i is before the first interval point
        if interval_seq and i < interval_seq[0]:
            # print('pipeline 88 i is before the first interval point:', i, flush=True)
            return list(range(i, min(i + max_parallel, interval_seq[0])))
            
        return []

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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )

        # Store multi-GPU models
        self.multi_gpu_denoising_unets = multi_gpu_denoising_unets
        self.gpu_ids = gpu_ids
        self.max_parallel_per_gpu = max_parallel_per_gpu

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

        assert attention_mask is None

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


    def process_on_gpu(
        self,
        gpu_id,
        sample: torch.FloatTensor,
        t_batch,
        encoder_hidden_states: torch.Tensor,
        audio_cond_fea: Optional[torch.Tensor] = None,
        face_musk_fea: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # save
        save_dir: Optional[str] = '',
        save_layers: Optional[List[str]] = [],
        save_types: Optional[List[str]] = [],
        timestep_idx: Optional[int] = 0,
        # DeepCache params
        prv_features: Optional[torch.Tensor] = None,
        quick_replicate: bool = False,
        cache_layer_id: Optional[int] = None,
        cache_block_id: Optional[int] = None,
        # masked attention
        mask_fg: Optional[torch.Tensor] = None,
        use_masked_attention: bool = False,
        masked_layers: Optional[Dict[str, List[str]]] = {'spatial': [], 'audio': [], 'temporal': []},
        remove_ref_list: Optional[List[str]] = [],
        # bg cache
        bg_features: Optional[Dict[str, torch.Tensor]] = None,
        use_cached_bg: bool = False,
        # superfast cache
        superfast_replicate: bool = False,
        result_dict: Optional[dict] = None,
        batch_idx: Optional[int] = None,
        # test efficiency
        log_file: Optional[str] = None,
        log_flops: bool = False,
        log_latency: bool = False,
    ):
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


        # # Package all inputs in a dictionary
        inputs = {
            'sample': sample,
            't_batch': t_batch,
            'encoder_hidden_states': encoder_hidden_states,
            'audio_cond_fea': audio_cond_fea,
            'face_musk_fea': face_musk_fea,
            'replicate_prv_feature': prv_features,
            'mask_fg': mask_fg,
            'bg_features': bg_features
        }

        # Move all inputs to the device at once
        if log_latency:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        if gpu_id != 0:
            inputs = move_tensors_to_device(inputs, device)
        if log_latency:
            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event)
            print(f"moving tensor to gpu {gpu_id} took {latency} ms", flush=True)

        # print('video_pipeline 293 remove_ref_list:', remove_ref_list, flush=True)

        # print('video_pipeline 293 sample.shape:', inputs['sample'].shape, flush=True)

        if log_latency:
            start_event.record()
        # Process on this GPU
        with torch.no_grad(), thread_flops_counter:
            noise_pred, new_prv_features = denoising_unet(
                sample=inputs['sample'],
                timestep=inputs['t_batch'],
                encoder_hidden_states=inputs['encoder_hidden_states'],
                audio_cond_fea=inputs['audio_cond_fea'],
                face_musk_fea=inputs['face_musk_fea'],
                return_dict=return_dict,
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

        if log_latency:
            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event)
            print(f"gpu {gpu_id} denoising unet took {latency} ms", flush=True)

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
        # save
        save_dir: Optional[str] = '',
        save_layers: Optional[List[str]] = [],
        save_types: Optional[List[str]] = [],
        # DeepCache params
        cache_interval: int = 1,
        cache_layer_id: int = None,
        cache_block_id: int = None,
        uniform: bool = True,
        pow: float = None,
        interval_seq: List[int] = [],
        # parallel
        estimate_input_latents: bool = False,
        # ours
        mask_fg=None, # added: {64: mask_64, 32: mask_32}
        bg_cache_interval: int = 1,
        use_masked_attention: bool = False,
        masked_layers: List[int] = {'spatial': [], 'audio': [], 'temporal': []},
        remove_ref_list: List[int] = None,
        acc: bool = False,
        # test efficiency
        log_file: Optional[str] = None,
        log_flops: bool = False,
        log_latency: bool = False,
        **kwargs,
    ):
        # print('acc:', acc, flush=True)
        # assert 0
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

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

        if interval_seq is None:
            if cache_interval == 1:
                interval_seq = list(range(num_inference_steps))
            else:
                interval_seq = list(range(0, num_inference_steps, cache_interval))
     
        interval_seq = sorted(interval_seq)
        print('interval_seq:', interval_seq)

        # prv_features = None

        # important_indices = {}
        if bg_cache_interval > 1:
            full_bg_features = {} # bg features of all contexts
            if do_classifier_free_guidance:
                full_bg_features['uncond'] = {}
            full_bg_features['cond'] = {}
        else:
            full_bg_features = None

        if bg_cache_interval == 1:
            interval_seq_bg = list(range(num_inference_steps))
        else:
            interval_seq_bg = list(range(0, num_inference_steps, bg_cache_interval))
                # assert 0

        interval_seq_bg = sorted(interval_seq_bg)
        interval_seq_bg = interval_seq
        # interval_seq_bg = [0, 1, 2 ,3, 6, 9, 12, 17, 24, 27]
        # interval_seq_bg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 24, 27]

        if log_latency:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        if log_flops:
            flops_counter = FlopCounterMode(self.denoising_unet, depth=None, display=False)
        else:
            flops_counter = nullcontext()

        denoising_times = []
        flops_list = []

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
        print("ref_image_latents shape:", ref_image_latents.shape)
        print("face_mask_tensor shape:", face_mask_tensor.shape)
        print("face_locator_tensor shape:", face_locator_tensor.shape)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            skip_steps = 0
            prv_noise = None # for estimating input latent 

            if do_classifier_free_guidance:
                prv_features = torch.zeros(2, 320, latents.shape[2], 64, 64).to(device, dtype=latents.dtype)
            else:
                prv_features = torch.zeros(1, 320, latents.shape[2], 64, 64).to(device, dtype=latents.dtype)

            latency_count = 0

            for t_i, t in enumerate(timesteps):
                if skip_steps > 0:
                    skip_steps -= 1
                    if t_i == len(timesteps) - 1 or (t_i + 1) > num_warmup_steps and (t_i + 1) % self.scheduler.order == 0:
                        progress_bar.update()
                        if callback is not None and t_i % callback_steps == 0:
                            step_idx = t_i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
                    # denoising_times.append(0.0)
                    # print('line 691 should continue')
                    continue

                num_gpus = len(self.gpu_ids) if self.gpu_ids else 1
                parallel_timesteps = []
                # print('pipeline 687 t_i:', t_i, flush=True)
                # print('pipeline 688 interval_seq:', interval_seq, flush=True)
                # print("pipeline 689 self.max_parallel_per_gpu*num_gpus:", self.max_parallel_per_gpu*num_gpus, flush=True)
                
                if t_i-1 in interval_seq and interval_seq.index(t_i-1) != len(interval_seq)-1:
                    parallel_indices = get_next_indices(interval_seq, t_i-1, max_parallel=self.max_parallel_per_gpu*num_gpus, num_total=num_inference_steps)
                elif t_i not in interval_seq and t_i < num_inference_steps-1:
                    parallel_indices = get_next_indices(interval_seq, t_i, max_parallel=self.max_parallel_per_gpu*num_gpus, num_total=num_inference_steps)
                else:
                    parallel_indices = []

                print('t_i:', t_i, flush=True)
                print('parallel_indices:', parallel_indices, flush=True)
                # assert 0
                parallel_timesteps = [timesteps[j] for j in parallel_indices]

                parallel_t = [timesteps[t_i]]
                if len(parallel_timesteps)>1:
                    parallel_t=parallel_timesteps
                #     print('parallel_t:', parallel_t, flush=True)
                print('parallel_t:', parallel_t, flush=True)

                # Limit batch size to available GPUs * max_parallel
                num_timesteps = len(parallel_t)
                max_batch_size = num_gpus * self.max_parallel_per_gpu   # max_parallel=3 from your code
                if num_timesteps > max_batch_size:
                    # Truncate to max batch size
                    parallel_t = parallel_t[:max_batch_size]
                    num_timesteps = len(parallel_t)

                if t_i in interval_seq:
                    # prv_features = None   
                    # prv_features = {c[0]: None for c in context_queue}
                    prv_features = prv_features.zero_()
                    use_cached_prv_feature = False
                else:
                    use_cached_prv_feature = True

                if t_i in interval_seq_bg:
                    use_cached_bg = False
                else:
                    use_cached_bg = True
                
                if len(parallel_t) > 1:
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0]*len(parallel_t) * (2 if do_classifier_free_guidance else 1),
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
                
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                
                # print('self.denoising_unet.reference_unet.:', self.denoising_unet.reference_unet)
                # 1. Forward reference image
                if t_i == 0:
                    # print("do_classifier_free_guidance:", do_classifier_free_guidance)
                    # assert 0
                    self.reference_unet(
                        ref_image_latents,
                        torch.zeros_like(t),
                        encoder_hidden_states=None,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer, do_classifier_free_guidance=do_classifier_free_guidance, device=self.denoising_unet.device)
                    if self.multi_gpu_denoising_unets is not None:
                        # print('self.multi_gpu_denoising_unets.keys():', self.multi_gpu_denoising_unets.keys(), flush=True)
                        for gpu_id, gpu_control in reference_controls.items():
                            # print('gpu_id:', gpu_id, flush=True)
                            gpu_control.update(reference_control_writer, do_classifier_free_guidance=do_classifier_free_guidance, device=self.multi_gpu_denoising_unets[gpu_id].device)


                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                global_context = []
                for j in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            j * context_batch_size : (j + 1) * context_batch_size
                        ]
                    )
                
                # print('len(global_context):', len(global_context), flush=True)
                # print('num_inference_steps:', num_inference_steps, flush=True)
                # assert 0
                
                for context in global_context:
                    # print('context:', context, flush=True)
                    # assert 0

                    new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
                    for c_j in range(len(context)):
                        for c_i in range(len(context[c_j])):
                            new_context[c_j][c_i] = (context[c_j][c_i] + t_i * 2) % video_length

                    if full_bg_features is None:
                        bg_features = None
                    elif t_i == 0:
                        bg_features = {}
                    else:
                        bg_features = {}
                        assert len(new_context) == 1
                        for key in full_bg_features['cond'].keys():
                            # print('pipeline 855 preparing key {} for context: {}'.format(key, context), flush=True)
                            for c in new_context:
                                # new_c = get_bg_features_context(c, do_classifier_free_guidance, latents.shape[2])
                                # print('pipeline 853 c:', c, flush=True)
                                if do_classifier_free_guidance:
                                    bg_uncond = full_bg_features['uncond'][key][c]
                                    bg_cond = full_bg_features['cond'][key][c]
                                    bg_features[key] = torch.cat([bg_uncond, bg_cond], 0)
                                else:
                                    bg_features[key] = full_bg_features['cond'][key][c]
                
                    if estimate_input_latents:
                        # print('later timesteps: use different latent model input for each parallel timestep')
                        latent_model_input_list = [latents]
                        latent_model_input = latents
                        
                        for m in range(len(parallel_t)-1):
                            latent_model_input = self.scheduler.step(prv_noise, parallel_t[m], latent_model_input, **extra_step_kwargs, return_dict=False)[0]
                            latent_model_input_list.append(latent_model_input)
                        
                        # self.scheduler._step_index = self.scheduler._step_index - (len(parallel_t)-1)
                        # print('len(latent_model_input_list):', len(latent_model_input_list), flush=True)
                        latent_model_input = torch.cat(latent_model_input_list, dim=0)
                      
                        latent_model_input = (
                            torch.cat([latent_model_input[:, :, c] for c in new_context])
                            .to(device)
                            .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        )
                        # print('pipeline 850 latent_model_input.mean():', latent_model_input.mean(), flush=True)
                        assert latent_model_input.shape[0] == 2*len(parallel_t) if do_classifier_free_guidance else len(parallel_t)
                    else:
                        print('estimate_input_latents:', estimate_input_latents, flush=True)
                        latent_model_input = (
                            torch.cat([latents[:, :, c] for c in new_context])
                            .to(device)
                            .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        )
                        # print('pipeline 859 latent_model_input.mean():', latent_model_input.mean(), flush=True)
                        latent_model_input = latent_model_input.repeat(len(parallel_t), 1, 1, 1, 1)
                    
                    # print('do_classifier_free_guidance:', do_classifier_free_guidance, flush=True)
                    # assert 0

                    c_audio_latents = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)
                    audio_latents = torch.cat([torch.zeros_like(c_audio_latents), c_audio_latents], 0)

                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # print('pipeline 792 latent_model_input.mean():', latent_model_input.mean(), flush=True)
                    # assert 0

                    if use_cached_prv_feature:
                        replicate_prv_feature = torch.cat([prv_features[:,:,c] for c in new_context])
                    else:
                        replicate_prv_feature = None

                    if log_latency:
                        start_event.record()

                    if len(parallel_t) == 1 or self.multi_gpu_denoising_unets is None:
                        # print('pipeline 799 latent_model_input shape:', latent_model_input.shape, flush=True)
                        # print('pipeline 915 t_i:', t_i, 'use_cached_bg:', use_cached_bg, flush=True)
                        
                        with flops_counter:
                            pred, prv_feature = self.denoising_unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=None,
                                audio_cond_fea=audio_latents if do_classifier_free_guidance else c_audio_latents,
                                face_musk_fea=face_locator_tensor if do_classifier_free_guidance else c_face_locator_tensor,
                                return_dict=False,
                                ############
                                mask_fg=mask_fg,
                                use_masked_attention=use_masked_attention,
                                use_cached_bg=use_cached_bg,
                                masked_layers=masked_layers,
                                remove_ref_list=remove_ref_list,
                                ############### DeepCache ###############
                                # replicate_prv_feature=torch.cat([prv_features[:,:,c] for c in new_context]) if prv_features is not None else None,
                                replicate_prv_feature=replicate_prv_feature,
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
                                timestep_idx=t_i,
                            )

                        if log_flops:
                            gflops = flops_counter.get_total_flops() / 1e9
                            print(f"Denoising UNet step took {gflops} GFLOPs", flush=True)
                            flops_list.append(gflops)

                    else:
                        # if latency_count == 90:
                        #     num_gpus = len(self.gpu_ids) - 1
                        # else:
                        #     num_gpus = len(self.gpu_ids)

                        # Determine how many batches per GPU
                        batches_per_gpu = [0] * num_gpus

                        for k in range(num_timesteps):
                            batches_per_gpu[k % num_gpus] += 1

                        # Prepare input batches for each GPU
                        gpu_batches = []
                        t_index = 0
                        # print('pipeline 880 latent_model_input.shape:', latent_model_input.shape, flush=True)
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
                        # print('pipeline 998 use_cached_bg:', use_cached_bg, flush=True)
                        for batch_idx, (gpu_idx, t_batch, latent_model_input_batch) in enumerate(gpu_batches):    
                            # print('pipeline 1026 batch_idx:', batch_idx, flush=True)
                            # print('pipeline 1027 gpu_idx:', gpu_idx, flush=True)
                            # print('pipeline 1028 t_batch:', t_batch, flush=True)
                            thread = threading.Thread(
                                target=self.process_on_gpu,
                                kwargs={
                                    'gpu_id': self.gpu_ids[gpu_idx], 
                                    'sample': latent_model_input_batch,
                                    't_batch': t_batch,
                                    'encoder_hidden_states': None,
                                    'audio_cond_fea': audio_latents if do_classifier_free_guidance else c_audio_latents,
                                    'face_musk_fea': face_locator_tensor if do_classifier_free_guidance else c_face_locator_tensor,
                                    'return_dict': False,
                                    'prv_features': replicate_prv_feature,
                                    'mask_fg': mask_fg,
                                    'use_masked_attention': use_masked_attention,
                                    'use_cached_bg': use_cached_bg,
                                    'masked_layers': masked_layers,
                                    'remove_ref_list': remove_ref_list,
                                    'quick_replicate': cache_interval > 1,  # quick_replicate
                                    'superfast_replicate': cache_block_id == -1,  # superfast_replicate
                                    'cache_layer_id': cache_layer_id,
                                    'cache_block_id': cache_block_id,
                                    'bg_features': bg_features,
                                    'save_dir': save_dir,
                                    'save_layers': save_layers,
                                    'save_types': save_types,
                                    'result_dict': results,
                                    'batch_idx': batch_idx,
                                    'log_file': log_file,
                                    'log_flops': log_flops,
                                    'log_latency': log_latency,
                                }
                            )
                            threads.append(thread)
                            thread.start()

                        # Wait for all threads to complete
                        for thread in threads:
                            thread.join()

                        # Combine results in the correct order
                        combined_noise_pred = []
                        gflops = 0
                        # print('pipeline 808 results.keys():', results.keys(), flush=True)
                        # print('pipeline 809 len(gpu_batches):', len(gpu_batches), flush=True)
                        for batch_idx in range(len(gpu_batches)):
                            batch_results = results[batch_idx]
                            combined_noise_pred.append(batch_results['noise_pred'])
                            gflops += batch_results['flops']

                        # Concatenate noise predictions
                        pred = torch.cat(combined_noise_pred, dim=0)
                        
                        # Use the last batch's prv_features (typically from GPU 0)
                        prv_feature = results[len(gpu_batches)-1]['prv_features']  

                        if log_flops:
                            print(f"Denoising UNet step took {gflops} GFLOPs", flush=True)
                            flops_list.append(gflops)

                    if t_i in interval_seq_bg and bg_features is not None:
                        # print('pipeline 954 bg_features.keys():', bg_features.keys(), flush=True)
                        # print('latents.shape[2]:', latents.shape[2], flush=True)
                        for key in bg_features.keys():
                            assert len(new_context) == 1
                            # print('pipeline 958 key: {}, value.shape: {}'.format(key, bg_features[key].shape), flush=True)
                            for c in new_context:
                                if key not in full_bg_features['cond']:
                                    # print('pipeline 971 initializing keys', key, flush=True)
                                    if do_classifier_free_guidance:
                                        full_bg_features['uncond'][key] = torch.zeros(
                                            (latents.shape[2], 
                                            *bg_features[key].shape[1:]),
                                            device=bg_features[key].device,
                                            dtype=bg_features[key].dtype
                                        )
                                    full_bg_features['cond'][key] = torch.zeros(
                                        (latents.shape[2], 
                                        *bg_features[key].shape[1:]),
                                        device=bg_features[key].device,
                                        dtype=bg_features[key].dtype
                                    )
                                
                                # print('pipeline 987 updating key {} for context: {}'.format(key, c), flush=True)
                                if do_classifier_free_guidance:
                                    bs = bg_features[key].shape[0] // 2
                                    full_bg_features['uncond'][key][c] = bg_features[key][:bs]
                                    full_bg_features['cond'][key][c] = bg_features[key][bs:]
                                else:
                                    full_bg_features['cond'][key][c] = bg_features[key]

                    if log_latency:
                        print("latency_count:", latency_count, flush=True)
                            
                        end_event.record()
                        end_event.synchronize()
                        time_elapsed = start_event.elapsed_time(end_event) / 1000
                        print(f"Denoising UNet step {t_i} took {time_elapsed} seconds for current context", flush=True)
                        denoising_times.append(time_elapsed)

                        # if latency_count == 90:
                        #     print("context:", new_context, flush=True)
                        #     print("t_i:", t_i, flush=True)
                        #     print("parallel_t:", parallel_t, flush=True)
                        #     print("len(parallel_t):", len(parallel_t), flush=True)
                        
                        # if latency_count == 150:
                        #     assert 0

                        latency_count += 1

                    # if log_flops:
                    #     print(f"Denoising UNet step {t_i} took {gflops} GFLOPs for current context", flush=True)
                    #     flops_list.append(gflops)

                    # print('pipeline 985 pred.shape:', pred.shape, flush=True)
                    # if len(parallel_t) == 1:
                    if acc:
                        # print('pipeline 969 t:', t, flush=True)
                        alphas_cumprod = self.scheduler.alphas_cumprod.to(latent_model_input.device)
                        if len(parallel_t) > 1:
                            new_pred_list = []
                            for p_t in range(len(parallel_t)):
                                step_size = 2 if do_classifier_free_guidance else 1
                                x_pred = origin_by_velocity_and_sample(pred[p_t:p_t+step_size], latent_model_input[p_t:p_t+step_size], alphas_cumprod, parallel_t[p_t])
                                new_pred = psuedo_velocity_wrt_noisy_and_timestep(latent_model_input[p_t:p_t+step_size], x_pred, alphas_cumprod, parallel_t[p_t], torch.ones_like(parallel_t[p_t]) * (-1))
                                new_pred_list.append(new_pred)
                            
                            pred = torch.cat(new_pred_list, dim=0)
                            # print('pipeline 990 pred.shape:', pred.shape, flush=True)
                        else:
                            x_pred = origin_by_velocity_and_sample(pred, latent_model_input, alphas_cumprod, t)
                            pred = psuedo_velocity_wrt_noisy_and_timestep(latent_model_input, x_pred, alphas_cumprod, t, torch.ones_like(t) * (-1))
                            # print('pipeline 994 pred.shape:', pred.shape, flush=True)
                        # print('pipeline 934 x_pred.mean():', x_pred.mean(), flush=True)
                        # print('pipeline 935 pred.mean():', pred.mean(), flush=True)
                        
                
                    for j, c in enumerate(new_context):
                        # print('pipeline 970 noise_pred[:, :, c].shape:', noise_pred[:, :, c].shape, flush=True)
                        # print('pipeline 971 pred.shape:', pred.shape, flush=True)
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        if prv_feature is not None:
                            prv_features[:, :, c] = prv_feature 

                        counter[:, :, c] = counter[:, :, c] + 1
                    
                    # break
                    
                # perform guidance
                if do_classifier_free_guidance:
                    if len(parallel_t) > 1:
                        noise_pred_uncond = noise_pred[::2]
                        noise_pred_text = noise_pred[1::2]
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    noise_pred = noise_pred / counter
                else:
                    noise_pred = noise_pred / counter # acc
                    # print('pipeline 953 noise_pred.mean():', noise_pred.mean(), flush=True)

                for j, timestep in enumerate(parallel_t):
                    latents = self.scheduler.step(
                        noise_pred[j].unsqueeze(0), timestep, latents, **extra_step_kwargs
                    ).prev_sample
                
                prv_noise = noise_pred[-1].unsqueeze(0)
                # assert 0
                # print('pipeline 961 latents.mean():', latents.mean(), flush=True)
                # assert 0
                if len(parallel_t) > 1:
                    skip_steps = len(parallel_t) - 1            

                if t_i == len(timesteps) - 1 or (
                    (t_i + 1) > num_warmup_steps and (t_i + 1) % self.scheduler.order == 0
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
                        f.write(f'timestep, context {i}: {denoising_time}\n')
                    # Calculate and print the average denoising time
                    avg_denoising_time = sum(denoising_times) / num_inference_steps / len(global_context)
                    f.write(f"Average denoising time per step per context: {avg_denoising_time} seconds\n")
            if log_flops:
                with open(log_file, 'a') as f:
                    for i, gflops in enumerate(flops_list):
                        f.write(f'timestep, context {i}: {gflops}\n')
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