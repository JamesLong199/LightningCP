import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

import time
import threading

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
class VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    denoising_times: Optional[List[float]] = None


class VideoPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae,
        reference_net,
        diffusion_net,
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
            reference_net=reference_net,
            diffusion_net=diffusion_net,
            scheduler=scheduler,
            image_proj=image_proj,
        )

        self.vae_scale_factor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
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
        batch_size: int,  # Number of videos to generate in parallel
        num_channels_latents: int,  # Number of channels in the latents
        width: int,  # Width of the video frame
        height: int,  # Height of the video frame
        video_length: int,  # Length of the video in frames
        dtype: torch.dtype,  # Data type of the latents
        device: torch.device,  # Device to store the latents on
        generator: Optional[torch.Generator] = None,  # Random number generator for reproducibility
        latents: Optional[torch.Tensor] = None,  # Pre-generated latents (optional)
    ):
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
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def process_on_gpu(
        self,
        gpu_id,
        sample: torch.FloatTensor,
        ref_features: dict,
        t_batch,
        # timestep: Union[torch.Tensor, float, int, list],
        encoder_hidden_states: torch.Tensor,
        audio_embedding: Optional[torch.Tensor] = None,
        audio_emotion: Optional[torch.Tensor] = None,
        # class_labels: Optional[torch.Tensor] = None,
        # mask_cond_fea: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        # mid_block_additional_residual: Optional[torch.Tensor] = None,
        uc_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        is_new_audio=True,
        update_past_memory=False,
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
        masked_layers: Optional[dict[str, list[str]]] = {'spatial': [], 'audio': [], 'temporal': []},
        remove_ref_list: Optional[list[str]] = [],
        # bg cache
        bg_features: Optional[dict[str, torch.Tensor]] = None,
        use_cached_bg: bool = False,
        # superfast cache
        superfast_replicate: bool = False,
        result_dict: Optional[dict] = None,
        batch_idx: Optional[int] = None,
        # test efficiency
        log_file=None,
        log_flops=False,
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
            denoising_unet = self.diffusion_net
        else:
            denoising_unet = self.multi_gpu_denoising_unets[gpu_id]

        # Set up FLOP counter for this thread if needed
        if log_flops:
            thread_flops_counter = FlopCounterMode([denoising_unet], depth=None, display=False)
        else:
            thread_flops_counter = nullcontext()

        # # Package all inputs in a dictionary
        inputs = {
            'latent_model_input': sample,
            'ref_features': ref_features,
            't_batch': t_batch,
            'encoder_hidden_states': encoder_hidden_states,
            'audio_embedding': audio_embedding,
            'audio_emotion': audio_emotion,
            'uc_mask': uc_mask,
            'replicate_prv_feature': prv_features,
            'mask_fg': mask_fg,
            'bg_features': bg_features
        }

        # Move all inputs to the device at once
        if gpu_id != 0:
            inputs = move_tensors_to_device(inputs, device)

        print('video_pipeline 293 remove_ref_list:', remove_ref_list, flush=True)

        # Process on this GPU
        with torch.no_grad(), thread_flops_counter:
            noise_pred, new_prv_features = denoising_unet(
                sample=inputs['latent_model_input'],
                ref_features=inputs['ref_features'],
                timestep=inputs['t_batch'],
                encoder_hidden_states=inputs['encoder_hidden_states'],
                audio_embedding=inputs['audio_embedding'],
                audio_emotion=inputs['audio_emotion'],
                uc_mask=inputs['uc_mask'],
                is_new_audio=is_new_audio,
                update_past_memory=update_past_memory,
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
            'noise_pred': noise_pred.to(self.diffusion_net.device),
            'prv_features': new_prv_features.to(self.diffusion_net.device),
            'flops': thread_flops,
        }


    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        face_emb,
        audio_tensor,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        audio_emotion=None,
        emotion_class_num=None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        is_new_audio: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
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
        # ours
        mask_fg=None, # added: {64: mask_64, 32: mask_32}
        bg_cache_interval: int = 1,
        use_masked_attention: bool = False,
        masked_layers: List[int] = {'spatial': [], 'audio': [], 'temporal': []},
        remove_ref_list: List[int] = [],
        estimate_input_latents: bool = False,
        ############# test efficiency ############
        log_file=None,
        log_latency=False,
        log_flops=False,     
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # print(f'video_pipeline 176 timesteps: {timesteps}')
        # assert 0

        batch_size = 1

        # prepare clip image embeddings
        clip_image_embeds = face_emb
        clip_image_embeds = clip_image_embeds.to(self.image_proj.device, self.image_proj.dtype)

        encoder_hidden_states = self.image_proj(clip_image_embeds)
        uncond_encoder_hidden_states = self.image_proj(torch.zeros_like(clip_image_embeds))

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

        num_channels_latents = self.diffusion_net.in_channels

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

        # DeepCache
        if cache_interval == 1:
            interval_seq = list(range(num_inference_steps))
        else:
            if uniform:
                interval_seq = list(range(0, num_inference_steps, cache_interval))
            else:
                num_slow_step = num_inference_steps//cache_interval
                if num_inference_steps%cache_interval != 0:
                    num_slow_step += 1

        interval_seq = sorted(interval_seq)
        prv_features = None
        # interval_seq = [0, 1, 2, 4, 7, 12, 16, 18] # non-uniform 3
        # interval_seq = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18] # non-uniform 1
        # interval_seq = [0, 1, 2, 3, 6, 9, 12, 15, 18] # non-uniform 1
        interval_seq = [0, 1, 2, 3, 6, 12, 15, 18] # non-uniform 1

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = rearrange(ref_image, "b f c h w -> (b f) c h w")
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image_tensor, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        # To save memory on GPUs like RTX 4090, we encode each frame separately
        # ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = []
        for frame_idx in range(ref_image_tensor.shape[0]):
            ref_image_latents.append(self.vae.encode(ref_image_tensor[frame_idx : frame_idx + 1]).latent_dist.mean)
        ref_image_latents = torch.cat(ref_image_latents, dim=0)

        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        if do_classifier_free_guidance:
            uncond_audio_tensor = torch.zeros_like(audio_tensor)
            audio_tensor = torch.cat([uncond_audio_tensor, audio_tensor], dim=0)
            audio_tensor = audio_tensor.to(dtype=self.diffusion_net.dtype, device=self.diffusion_net.device)

        if log_latency:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        if log_flops:
            flops_counter = FlopCounterMode(self.diffusion_net, depth=None, display=False)
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
                #interval_seq, pow = sample_from_quad(num_inference_steps, num_inference_steps//cache_interval, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                # print(interval_seq_bg, len(interval_seq_bg), pow)
                # assert 0

        # print('cache_interval:', cache_interval)
        interval_seq_bg = sorted(interval_seq_bg)

        interval_seq_bg = interval_seq
        print('interval_seq_bg:', interval_seq_bg, flush=True)


        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        skip_steps = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps)):

                if skip_steps > 0:
                    skip_steps -= 1
                    if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
                    denoising_times.append(0.0)
                    # print('line 691 should continue')
                    continue

                if i in interval_seq:
                    prv_features = None      
                    # bg_features = None
                if i in interval_seq_bg:
                    use_cached_bg = False
                else:
                    use_cached_bg = True
                print('use_cached_bg:', use_cached_bg, flush=True)

                num_gpus = len(self.gpu_ids) if self.gpu_ids else 1
                parallel_timesteps = []
                if i-1 in interval_seq:
                    parallel_indices = get_next_indices(interval_seq, i-1, max_parallel=self.max_parallel_per_gpu*num_gpus)
                elif i not in interval_seq:
                    parallel_indices = get_next_indices(interval_seq, i, max_parallel=self.max_parallel_per_gpu*num_gpus)
                else:
                    parallel_indices = []

                print('i:', i, flush=True)
                print('parallel_indices:', parallel_indices, flush=True)
                parallel_timesteps = [timesteps[j] for j in parallel_indices]

                t = timesteps[i]

                parallel_t = [t]
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

                # Forward reference image
                if i == 0:
                    ref_features = self.reference_net(
                        ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                        torch.zeros_like(t),
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if estimate_input_latents:
                    if i < 3 or i > 18:
                        # print('face animate 620 latent_model_input.shape:', latent_model_input.shape, flush=True)
                        print('early timesteps: use same latent model input for all parallel timesteps')
                        latent_model_input = latent_model_input.repeat(len(parallel_t), 1, 1, 1, 1)
                        # print('face animate 623 latent_model_input.shape:', latent_model_input.shape, flush=True)
                    else:
                        # latent_input = latent_model_input
                        print('later timesteps: use different latent model input for each parallel timestep')
                        latent_model_input_list = [latent_model_input]
                        
                        for m in range(len(parallel_t)-1):
                            # print('face animate 628 latent_model_input.shape:', latent_model_input.shape, flush=True)
                            # print('face animate 631 noise_pred.shape:', noise_pred.shape, flush=True)
                            latent_model_input = self.scheduler.step(noise_pred[-1].unsqueeze(0), parallel_t[m], latent_model_input, **extra_step_kwargs, return_dict=False)[0]
                            # print('video_pipeline 596 self.scheduler._step_index:', self.scheduler._step_index, flush=True)
                            latent_model_input_list.append(latent_model_input)
                        
                        self.scheduler._step_index = self.scheduler._step_index - (len(parallel_t)-1)
                        latent_model_input = torch.cat(latent_model_input_list, dim=0)
                        # print('video_pipeline 596 self.scheduler._step_index:', self.scheduler._step_index, flush=True)
                        # np.save(f'latent_model_input_parallel.npy', latent_model_input.detach().cpu().numpy())
                        # assert 0
                        assert latent_model_input.shape[0] == 2*len(parallel_t) if do_classifier_free_guidance else len(parallel_t)
                else:
                    print('estimate_input_latents:', estimate_input_latents, flush=True)
                    latent_model_input = latent_model_input.repeat(len(parallel_t), 1, 1, 1, 1)

                audio_emotion = torch.tensor(torch.mode(audio_emotion).values.item()).to(
                    dtype=torch.int, device=self.diffusion_net.device
                )
                if do_classifier_free_guidance:
                    uncond_audio_emotion = torch.full_like(audio_emotion, emotion_class_num)
                    audio_emotion = torch.cat(
                        [uncond_audio_emotion.unsqueeze(0), audio_emotion.unsqueeze(0)],
                        dim=0,
                    )

                    uc_mask = (
                        torch.Tensor(
                            [1] * batch_size * num_images_per_prompt * 16
                            + [0] * batch_size * num_images_per_prompt * 16
                        )
                        .to(device)
                        .bool()
                    )
                else:
                    uc_mask = None

                if log_latency:
                    start_event.record()

                if len(parallel_t) == 1 or self.multi_gpu_denoising_unets is None:
                    with flops_counter:
                        noise_pred, prv_features = self.diffusion_net(
                            latent_model_input,
                            ref_features,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            audio_embedding=audio_tensor,
                            audio_emotion=audio_emotion,
                            uc_mask=uc_mask,
                            is_new_audio=is_new_audio,
                            update_past_memory=i == 0,
                            return_dict=False,
                            # save
                            save_dir=save_dir,
                            save_layers=save_layers,
                            save_types=save_types,
                            timestep_idx=i,
                            ############### DeepCache ###############
                            replicate_prv_feature=prv_features,
                            quick_replicate= cache_interval>1,
                            superfast_replicate=cache_block_id==-1,
                            cache_layer_id=cache_layer_id,
                            cache_block_id=cache_block_id,
                            ############ ours ############
                            mask_fg=mask_fg,
                            use_masked_attention=use_masked_attention,
                            use_cached_bg=use_cached_bg,
                            masked_layers=masked_layers,
                            remove_ref_list=remove_ref_list,
                            ############### BG Cache ###############
                            bg_features=bg_features,
                        )#.sample

                    # print('video_pipeline 317 noise_pred.shape:', noise_pred.shape, flush=True)
                    # print('video_pipeline 318 prv_features.shape:', prv_features.shape, flush=True)

                    if log_flops:
                        gflops = flops_counter.get_total_flops() / 1e9
            
                else:

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

                    print('video_pipeline 663 remove_ref_list:', remove_ref_list, flush=True)

                    for batch_idx, (gpu_idx, t_batch, latent_model_input_batch) in enumerate(gpu_batches):    
                        print('video_pipeline 667 cache_interval:', cache_interval, flush=True)
                        print('video_pipeline 668 cache_block_id:', cache_block_id, flush=True)
                        if prv_features is not None:
                            print('video_pipeline 670 prv_features is not None', flush=True)
                        else:
                            print('video_pipeline 672 prv_features is None', flush=True)
                        thread = threading.Thread(
                            target=self.process_on_gpu,
                            kwargs={
                                'gpu_id': self.gpu_ids[gpu_idx], 
                                'sample': latent_model_input_batch,
                                'ref_features': ref_features,
                                't_batch': t_batch,
                                'encoder_hidden_states': encoder_hidden_states,
                                'audio_embedding': audio_tensor,
                                'audio_emotion': audio_emotion,
                                'uc_mask': uc_mask,
                                'is_new_audio': is_new_audio,
                                'update_past_memory': i==0, # update_past_memory
                                'prv_features': prv_features,
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
                    # print('face_animate 808 results.keys():', results.keys(), flush=True)
                    # print('face_animate 809 len(gpu_batches):', len(gpu_batches), flush=True)
                    for batch_idx in range(len(gpu_batches)):
                        batch_results = results[batch_idx]
                        combined_noise_pred.append(batch_results['noise_pred'])
                        gflops += batch_results['flops']

                    # Concatenate noise predictions
                    noise_pred = torch.cat(combined_noise_pred, dim=0)
                    
                    # Use the last batch's prv_features (typically from GPU 0)
                    prv_features = results[len(gpu_batches)-1]['prv_features']  

                    # if log_flops:
                    #     print(f"Denoising UNet step took {gflops} GFLOPs", flush=True)
                    #     flops_list.append(gflops)

                if log_latency:
                    end_event.record()
                    end_event.synchronize()
                    time_elapsed = start_event.elapsed_time(end_event) / 1000
                    print(f"Denoising UNet step took {time_elapsed} seconds", flush=True)
                    denoising_times.append(time_elapsed)

                if log_flops:
                    print(f"Denoising UNet step took {gflops} GFLOPs", flush=True)
                    flops_list.append(gflops)

                # perform guidance
                if do_classifier_free_guidance:
                    if len(parallel_t) > 1:
                        noise_pred_uncond = noise_pred[::2]
                        noise_pred_text = noise_pred[1::2]
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                for j, timestep in enumerate(parallel_t):
                    latents = self.scheduler.step(noise_pred[j].unsqueeze(0), timestep, latents, **extra_step_kwargs, return_dict=False)[0]

                if len(parallel_t) > 1:
                    skip_steps = len(parallel_t) - 1

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

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

        return VideoPipelineOutput(videos=images, denoising_times=denoising_times)
