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
        interval_seq: List[int] = [],
        # ours
        mask_fg=None, # added: {64: mask_64, 32: mask_32}
        bg_cache_interval: int = 1,
        use_masked_attention: bool = False,
        masked_layers: List[int] = {'spatial': [], 'audio': [], 'temporal': []},
        remove_ref_list: List[int] = None,
        video_name: Optional[str] = None,
        clip_idx: Optional[int] = None,
        # test efficiency
        log_file: Optional[str] = None,
        log_flops: bool = False,
        log_latency: bool = False,
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

        interval_seq = sorted(interval_seq)
        
        # print('interval_seq:', interval_seq, flush=True)
        
        prv_features = None
        # interval_seq = [0, 1, 2, 4, 7, 12, 16, 18] # non-uniform 3
        # interval_seq = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18] # non-uniform 1
        # interval_seq = [0, 1, 2, 3, 6, 9, 12, 15, 18] # non-uniform 1

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

        # assert 0

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps)):
                if i in interval_seq:
                    prv_features = None      
                    # bg_features = None
                if i in interval_seq_bg:
                    use_cached_bg = False
                else:
                    use_cached_bg = True
                # print('use_cached_bg:', use_cached_bg, flush=True)

                t = timesteps[i]
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
                        # superfast_replicate=superfast_replicate,
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

                if log_latency:
                    end_event.record()
                    end_event.synchronize()
                    time_elapsed = start_event.elapsed_time(end_event) / 1000
                    print(f"Denoising UNet step {i} took {time_elapsed} seconds", flush=True)
                    denoising_times.append(time_elapsed)

                if log_flops:
                    gflops = flops_counter.get_total_flops() / 1e9
                    print(f"Denoising UNet step {i} took {gflops} GFLOPs", flush=True)
                    flops_list.append(gflops)
                # print('video_pipeline 317 noise_pred.shape:', noise_pred.shape, flush=True)
                # print('video_pipeline 318 prv_features.shape:', prv_features.shape, flush=True)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # import numpy as np
                # import os
                # from memo.models.masking_utils import reshape_attn_data
                # os.makedirs(f'noise_pred_data/t{i:02d}', exist_ok=True)
                # np.save(f'noise_pred_data/t{i:02d}/t{i:02d}_noise_pred.npy', noise_pred[0].detach().cpu().float().numpy())

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # import numpy as np
                # import os
                # from memo.models.masking_utils import reshape_attn_data
                # os.makedirs(f'latents_data/t{i:02d}', exist_ok=True)
                # np.save(f'latents_data/t{i:02d}/t{i:02d}_latents.npy', reshape_attn_data(latents).detach().cpu().float().numpy())

                # with open(f'magnitude.txt', 'a') as f:
                #     f.write(f't{i:02d}: {torch.linalg.norm(latents)}\n')

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if save_dir is not None and video_name is not None and clip_idx is not None:
                    from attn_scripts.compute_attn_stats import compute_attention_statistics, process_multiple_files
                    # INSERT_YOUR_CODE
                    import glob
                    attn_files = glob.glob(f'attention_weights/t{i:02d}/*')
                    process_multiple_files(
                        file_paths=attn_files,
                        output_dir=f'stats/{video_name.split(".")[0]}/clip_{clip_idx}/attn_weight_stats/t{i:02d}',
                        save_format='npy',
                    )
                    # use os to delete data_dir
                    import shutil, os
                    if os.path.exists(f'attention_weights/t{i:02d}'):
                        shutil.rmtree(f'attention_weights/t{i:02d}')

                

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
