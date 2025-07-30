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

def find_memory_linear_attention_modules(model):
    """
    Find and print all modules that use MemoryLinearAttnProcessor.
    """
    memory_attn_modules = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'processor') and isinstance(module.processor, MemoryLinearAttnProcessor):
            memory_attn_modules.append(name)
            
    print(f"Found {len(memory_attn_modules)} modules with MemoryLinearAttnProcessor:")
    for name in memory_attn_modules:
        print(f"  {name}")
    
    return memory_attn_modules


# Add this to your code after loading the diffusion_net model
from memo.models.attention_processor import MemoryLinearAttnProcessor


def register_time(model, time):
    """
    Register current timestep in model.
    
    Args:
        model: UNet model
        time: Current timestep
    """
    setattr(model, 'order', time)

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

        # memory_modules = find_memory_linear_attention_modules(self.diffusion_net)
        # assert 0


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
        encoder_hidden_states: torch.Tensor,
        cond, # faster diffusion
        audio_embedding: Optional[torch.Tensor] = None,
        audio_emotion: Optional[torch.Tensor] = None,
        uc_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        is_new_audio=True,
        update_past_memory=False,
        result_dict: Optional[dict] = None,
        batch_idx: Optional[int] = None,
        mode: Optional[str] = 'full',
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
        }

        # Move all inputs to the device at once
        if gpu_id != 0:
            inputs = move_tensors_to_device(inputs, device)

        # Process on this GPU
        with torch.no_grad(), thread_flops_counter:
            noise_pred = denoising_unet(
                sample=inputs['latent_model_input'],
                ref_features=inputs['ref_features'],
                timestep=inputs['t_batch'],
                encoder_hidden_states=inputs['encoder_hidden_states'],
                cond=cond, # faster diffusion
                audio_embedding=inputs['audio_embedding'],
                audio_emotion=inputs['audio_emotion'],
                uc_mask=inputs['uc_mask'],
                is_new_audio=is_new_audio,
                update_past_memory=update_past_memory,
                return_dict=False,
                mode=mode, # multi-gpu
            )[0]

        # Calculate FLOPs for this thread
        thread_flops = 0
        if log_flops:
            thread_flops = thread_flops_counter.get_total_flops() / 1e9

        # Move results back to CPU to combine later
        result_dict[batch_idx] = {
            'noise_pred': noise_pred.to(self.diffusion_net.device),
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
        # faster diffusion
        prior_noise_injection: bool = False,
        cond: List[int] = [],
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



        # Forward reference image
    
        ref_features = self.reference_net(
            ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
            torch.zeros_like(timesteps[0]),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )

        if log_latency:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        if log_flops:
            flops_counter = FlopCounterMode(self.diffusion_net, depth=None, display=False)
        else:
            flops_counter = nullcontext()

        denoising_times = []
        flops_list = []

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        curr_step = 0
        all_steps = len(timesteps)
        num_gpus = len(self.gpu_ids)

       

        # Do this instead:
        def get_module_by_path(model, path):
            """Get a module by its path string"""
            parts = path.split('.')
            current = model
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            return current

        with self.progress_bar(total=num_inference_steps) as progress_bar:

            while curr_step < all_steps:
                register_time(self.diffusion_net, curr_step)

                # if curr_step == 0:
                if self.diffusion_net.order == 0:
                    update_past_memory = True
                else:
                    update_past_memory = False

                print('curr_step:', curr_step, flush=True)
                print('self.diffusion_net.order:', self.diffusion_net.order, flush=True)

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
                if hasattr(self.scheduler, "scale_model_input"):
                    print('video_pipeline 382 self.scheduler:', self.scheduler, flush=True)
                    assert 0
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
                
                
                if len(time_ls) == 1 or self.multi_gpu_denoising_unets is None:
                    with flops_counter:
                        noise_pred = self.diffusion_net(
                            sample=latent_model_input,
                            ref_features=ref_features,
                            timestep=time_ls,
                            encoder_hidden_states=encoder_hidden_states,
                            cond=cond,
                            audio_embedding=audio_tensor,
                            audio_emotion=audio_emotion,
                            uc_mask=uc_mask,
                            is_new_audio=is_new_audio,
                            update_past_memory=update_past_memory,
                            return_dict=False,
                        )[0]

                    if log_flops:
                        gflops = flops_counter.get_total_flops() / 1e9
        
                else:
                    with flops_counter:
                        self.diffusion_net(
                            sample=latent_model_input,
                            ref_features=ref_features,
                            timestep=time_ls,
                            encoder_hidden_states=encoder_hidden_states,
                            cond=cond,
                            audio_embedding=audio_tensor,
                            audio_emotion=audio_emotion,
                            uc_mask=uc_mask,
                            is_new_audio=is_new_audio,
                            update_past_memory=update_past_memory,
                            return_dict=False,
                            mode='down_only',
                        )

                    if log_flops:
                        gflops_down = flops_counter.get_total_flops() / 1e9
                        print(f"UNet encoder took {gflops_down} GFLOPs", flush=True)
                    

                    for idx in self.multi_gpu_denoising_unets.keys():
                        self.multi_gpu_denoising_unets[idx].skip_feature = tuple(
                            t.to(self.multi_gpu_denoising_unets[idx].device) for t in self.diffusion_net.skip_feature
                        )
                        self.multi_gpu_denoising_unets[idx].toup_feature = self.diffusion_net.toup_feature.to(self.multi_gpu_denoising_unets[idx].device)
                        
                    # Determine how many batches per GPU
                    batches_per_gpu = [0] * num_gpus
                    for k in range(len(time_ls)):
                        batches_per_gpu[k % num_gpus] += 1

                    # Prepare input batches for each GPU
                    gpu_batches = []
                    t_index = 0
                    for gpu_idx, batch_size in enumerate(batches_per_gpu):
                        if batch_size > 0:
                            t_batch = time_ls[t_index:t_index+batch_size]
                            gpu_batches.append((gpu_idx, t_batch, latent_model_input))
                            t_index += batch_size

                    # Process each batch on its assigned GPU
                    threads = []
                    results = {}

                    for batch_idx, (gpu_idx, t_batch, latent_model_input_batch) in enumerate(gpu_batches):    
                        # only update in the first batch (corresponding to the first timestep)
                        if update_past_memory and batch_idx != 0:
                            update_past_memory = False
                            
                        thread = threading.Thread(
                            target=self.process_on_gpu,
                            kwargs={
                                'gpu_id': self.gpu_ids[gpu_idx], 
                                'sample': latent_model_input_batch,
                                'ref_features': ref_features,
                                't_batch': t_batch,
                                'encoder_hidden_states': encoder_hidden_states,
                                'cond': cond,
                                'audio_embedding': audio_tensor,
                                'audio_emotion': audio_emotion,
                                'uc_mask': uc_mask,
                                'is_new_audio': is_new_audio,
                                'update_past_memory': update_past_memory, # update_past_memory
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
                    for batch_idx in range(len(gpu_batches)):
                        batch_results = results[batch_idx]
                        combined_noise_pred.append(batch_results['noise_pred'])
                        if log_flops:
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

                # perform guidance
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
                        f.write(f'timestep: {denoising_time}\n')
                    # Calculate and print the average denoising time
                    avg_denoising_time = sum(denoising_times) / num_inference_steps
                    f.write(f"Average denoising time per step: {avg_denoising_time} seconds\n")
            if log_flops:
                with open(log_file, 'a') as f:
                    for i, gflops in enumerate(flops_list):
                        f.write(f'timestep: {gflops}\n')
                    # Calculate and print the average flops
                    avg_flops = sum(flops_list) / num_inference_steps
                    f.write(f"Average flops per step: {avg_flops} GFLOPs\n")
                    
        return VideoPipelineOutput(videos=images, denoising_times=denoising_times)
