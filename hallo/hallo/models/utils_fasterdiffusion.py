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
from typing import Dict, List, Optional, Tuple, Union
# from hallo.models.unet_3d_original import UNet3DConditionOutput, UNet3DConditionModel
from hallo.models.unet_3d import UNet3DConditionOutput, UNet3DConditionModel

from copy import deepcopy

from hallo.animate.face_animate import FaceAnimatePipelineOutput

def warpped_feature(sample, step):
    """
    Wrapper for expanding 3D features for parallel inference.
    
    Args:
        sample: batch_size*dim*frames*h*w tensor
        step: timestep span
        
    Returns:
        Expanded tensor for parallel processing
    """
    # bs, dim, f, h, w = sample.shape
    # uncond_fea, cond_fea = sample.chunk(2)
    # uncond_fea = uncond_fea.repeat(step, 1, 1, 1, 1)  # (step * bs//2) * dim * f * h * w
    # cond_fea = cond_fea.repeat(step, 1, 1, 1, 1)  # (step * bs//2) * dim * f * h * w
    # return torch.cat([uncond_fea, cond_fea])

    return sample.repeat(step, 1, 1, 1, 1)

def warpped_skip_feature(block_samples, step):
    """
    Wrapper for expanding skip features for parallel inference.
    
    Args:
        block_samples: List of tensors representing skip features
        step: timestep span
        
    Returns:
        Tuple of expanded tensors
    """
    down_block_res_samples = []
    for sample in block_samples:
        sample_expand = warpped_feature(sample, step)
        down_block_res_samples.append(sample_expand)
    return tuple(down_block_res_samples)

def warpped_text_emb(text_emb, step):
    """
    Wrapper for expanding text embeddings for parallel inference.
    
    Args:
        text_emb: batch_size*seq_len*dim tensor
        step: timestep span
        
    Returns:
        Expanded text embedding tensor
    """
    # bs, token_len, dim = text_emb.shape
    # uncond_fea, cond_fea = text_emb.chunk(2)
    # uncond_fea = uncond_fea.repeat(step, 1, 1)  # (step * bs//2) * seq_len * dim
    # cond_fea = cond_fea.repeat(step, 1, 1)  # (step * bs//2) * seq_len * dim
    # return torch.cat([uncond_fea, cond_fea])  # (step*bs) * seq_len * dim

    return text_emb.repeat(step, 1, 1)

def warpped_audio_tensor(audio_tensor, step):
    """
    Wrapper for expanding audio tensors for parallel inference.
    
    Args:
        audio_tensor: Audio feature tensor
        step: timestep span
        
    Returns:
        Expanded audio tensor
    """

    # bs, c, t, d = audio_tensor.shape
    # uncond_fea, cond_fea = audio_tensor.chunk(2)
    # uncond_fea = uncond_fea.repeat(step, 1, 1, 1)
    # cond_fea = cond_fea.repeat(step, 1, 1, 1)

    # return torch.cat([uncond_fea, cond_fea])

    return audio_tensor.repeat(step, 1, 1, 1)

def warpped_timestep(timesteps, bs):
    """
    Wrapper for expanding timesteps for parallel inference.
    
    Args:
        timesteps: List of timesteps e.g., [981, 961, 941]
        bs: Batch size
        
    Returns:
        Tensor of expanded timesteps
    """
    # semi_bs = bs // 2
    # ts = []
    # for timestep in timesteps:
    #     timestep = timestep[None]
    #     texp = timestep.expand(semi_bs)
    #     ts.append(texp)
    # timesteps = torch.cat(ts)
    # return timesteps.repeat(2, 1).reshape(-1)

    semi_bs = bs // 2
    ts = []
    for timestep in timesteps:
        timestep = timestep[None]
        texp = timestep.expand(semi_bs)
        ts.append(texp)
        ts.append(texp)
    timesteps = torch.cat(ts)
    return timesteps.reshape(-1) 

def warpped_face_mask(face_mask, step):
    """
    Wrapper for expanding face masks for parallel inference.
    
    Args:
        face_mask: Face mask tensor
        step: timestep span
        
    Returns:
        Expanded face mask tensor
    """
    # For 3D masks
    new_masks = []
    for mask in face_mask:
        uncond_mask, cond_mask = mask.chunk(2)
        uncond_mask = uncond_mask.repeat(step, 1)
        cond_mask = cond_mask.repeat(step, 1)
        new_mask = torch.cat([uncond_mask, cond_mask])
        new_masks.append(new_mask)
    return new_masks



def register_time(model, time):
    """
    Register current timestep in model.
    
    Args:
        model: UNet model
        time: Current timestep
    """
    setattr(model, 'order', time)

def register_parallel_pipeline(pipe, mod='50ls'):
    """
    Register a parallel pipeline for face animation.
    
    Args:
        pipe: FaceAnimatePipeline instance
        mod: Timestep modulation strategy
    """
    original_call = pipe.__call__
    
    def new_call(self):
        @torch.no_grad()
        def call(
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
            max_parallel_per_gpu: int = 3,
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
            
            # Main denoising loop with accelerated inference
            init_latents = latents.detach().clone()
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                while curr_step < all_steps:

                    register_time(self.denoising_unet, curr_step)
                    time_ls = [timesteps[curr_step]]
                    curr_step += 1
                    
                    # Collect timesteps until next key step
                    while curr_step not in cond:
                        if curr_step<all_steps and len(time_ls) < max_parallel_per_gpu:
                            time_ls.append(timesteps[curr_step])
                            curr_step += 1
                        else:
                            break

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Predict noise for all timesteps in parallel
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
            
            return FaceAnimatePipelineOutput(videos=images)
        
        return call
    pipe.call = new_call(pipe)

def register_faster_forward(model, mod = '50ls'):
    def faster_forward(self):
        def forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            cond: List[int],
            audio_embedding: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.Tensor] = None,
            mask_cond_fea: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_mask: Optional[torch.Tensor] = None,
            face_mask: Optional[torch.Tensor] = None,
            lip_mask: Optional[torch.Tensor] = None,
            motion_scale: Optional[torch.Tensor] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            # start: bool = False,
        ) -> Union[UNet3DConditionOutput, Tuple]:
            r"""
            Args:
                sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
                timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
                encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

            Returns:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
                returning a tuple, the first element is the sample tensor.
            """
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                logger.info(
                    "Forward upsample size to force interpolation output size.")
                forward_upsample_size = True

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # center input if necessary
            if self.config.center_input_sample:
                sample = 2 * sample - 1.0

            # time
            # faster diffusion
            if isinstance(timestep, list):
                timesteps = timestep[0]
                step = len(timestep)
                print('timestep:', timestep)
                print('step:', step)
            else:
                timesteps = timestep
                step = 1
            if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)): # faster diffusion
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor(
                    [timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # faster diffusion
            if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0])
            elif isinstance(timesteps, list):
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)
            t_emb = self.time_proj(timesteps)


            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.time_embedding(t_emb)

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError(
                        "class_labels should be provided when num_class_embeds > 0"
                    )

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb

            #===============
            order = self.order #timestep, start by 0
            #===============

            print("order:", order)
            if order in cond:
                # pre-process
                sample = self.conv_in(sample)
                if mask_cond_fea is not None:
                    sample = sample + mask_cond_fea

                # down
                down_block_res_samples = (sample,)
                for downsample_block in self.down_blocks:
                    if (
                        hasattr(downsample_block, "has_cross_attention")
                        and downsample_block.has_cross_attention
                    ):
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            full_mask=full_mask,
                            face_mask=face_mask,
                            lip_mask=lip_mask,
                            audio_embedding=audio_embedding,
                            motion_scale=motion_scale,
                        )
                        # print("")
                    else:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            # audio_embedding=audio_embedding,
                        )
                        # print("")

                    down_block_res_samples += res_samples

                if down_block_additional_residuals is not None:
                    new_down_block_res_samples = ()

                    for down_block_res_sample, down_block_additional_residual in zip(
                        down_block_res_samples, down_block_additional_residuals
                    ):
                        down_block_res_sample = (
                            down_block_res_sample + down_block_additional_residual
                        )
                        new_down_block_res_samples += (down_block_res_sample,)

                    down_block_res_samples = new_down_block_res_samples

                # mid
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    full_mask=full_mask,
                    face_mask=face_mask,
                    lip_mask=lip_mask,
                    audio_embedding=audio_embedding,
                    motion_scale=motion_scale,
                )

                if mid_block_additional_residual is not None:
                    sample = sample + mid_block_additional_residual

                #----------------------save feature-------------------------
                # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                setattr(self, 'skip_feature', deepcopy(down_block_res_samples))
                setattr(self, 'toup_feature', sample.detach().clone())
                #-----------------------save feature------------------------      

            else:
                down_block_res_samples = self.skip_feature
                sample = self.toup_feature

            #-------------------expand feature for parallel---------------
            if isinstance(timestep, list):
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timestep, sample.shape[0]).to(sample.device)
                t_emb = self.time_proj(timesteps)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb)
 
            down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
            sample = warpped_feature(sample, step)
            encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
            audio_embedding = warpped_audio_tensor(audio_embedding, step)
            full_mask = warpped_face_mask(full_mask, step)
            face_mask = warpped_face_mask(face_mask, step)
            lip_mask = warpped_face_mask(lip_mask, step)

            print('down_block_res_samples[0].shape after warpped_skip_feature:', down_block_res_samples[0].shape)
            print('sample.shape after warpped_feature:', sample.shape)
            print('encoder_hidden_states.shape after warpped_text_emb:', encoder_hidden_states.shape)
            print('')
            #-------------------expand feature for parallel---------------       
            
            print('############################# upblocks ##############################')
            # up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[
                    : -len(upsample_block.resnets)
                ]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if (
                    hasattr(upsample_block, "has_cross_attention")
                    and upsample_block.has_cross_attention
                ):
                    sample, _ = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        full_mask=full_mask,
                        face_mask=face_mask,
                        lip_mask=lip_mask,
                        audio_embedding=audio_embedding,
                        motion_scale=motion_scale,
                        num_parallel=step,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        encoder_hidden_states=encoder_hidden_states,
                        # audio_embedding=audio_embedding,
                    )

            # print('sample.shape before conv_norm_out:', sample.shape)
            # post-process
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            if not return_dict:
                return (sample,)

            return UNet3DConditionOutput(sample=sample)
        return forward
    if model.__class__.__name__ == 'UNet3DConditionModel':
        model.forward = faster_forward(model)




def register_faster_forward_multi_gpu(model, mod = '50ls'):
    def faster_forward(self):
        def forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            cond: List[int],
            audio_embedding: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.Tensor] = None,
            mask_cond_fea: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_mask: Optional[torch.Tensor] = None,
            face_mask: Optional[torch.Tensor] = None,
            lip_mask: Optional[torch.Tensor] = None,
            motion_scale: Optional[torch.Tensor] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            mode: str = 'full',
            # down_only: bool = False,
            # start: bool = False,
        ) -> Union[UNet3DConditionOutput, Tuple]:
            r"""
            Args:
                sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
                timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
                encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

            Returns:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
                returning a tuple, the first element is the sample tensor.
            """
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            assert mode in ['full', 'down_only', 'up_only']
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                logger.info(
                    "Forward upsample size to force interpolation output size.")
                forward_upsample_size = True

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # center input if necessary
            if self.config.center_input_sample:
                sample = 2 * sample - 1.0

            # time
            # faster diffusion
            if isinstance(timestep, list):
                timesteps = timestep[0]
                step = len(timestep)
                # print('timestep:', timestep)
                # print('step:', step)
            else:
                timesteps = timestep
                step = 1
            if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)): # faster diffusion
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor(
                    [timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # faster diffusion
            if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0])
            elif isinstance(timesteps, list):
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)
            t_emb = self.time_proj(timesteps)


            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.time_embedding(t_emb)

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError(
                        "class_labels should be provided when num_class_embeds > 0"
                    )

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb

            #===============
            order = self.order #timestep, start by 0
            #===============
            
            if mode == 'full' or mode == 'down_only':
                print("order:", order)
                if order in cond:
                    # pre-process
                    sample = self.conv_in(sample)
                    if mask_cond_fea is not None:
                        sample = sample + mask_cond_fea

                    # down
                    down_block_res_samples = (sample,)
                    for downsample_block in self.down_blocks:
                        if (
                            hasattr(downsample_block, "has_cross_attention")
                            and downsample_block.has_cross_attention
                        ):
                            sample, res_samples = downsample_block(
                                hidden_states=sample,
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                                full_mask=full_mask,
                                face_mask=face_mask,
                                lip_mask=lip_mask,
                                audio_embedding=audio_embedding,
                                motion_scale=motion_scale,
                            )
                            # print("")
                        else:
                            sample, res_samples = downsample_block(
                                hidden_states=sample,
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                # audio_embedding=audio_embedding,
                            )
                            # print("")

                        down_block_res_samples += res_samples

                    if down_block_additional_residuals is not None:
                        new_down_block_res_samples = ()

                        for down_block_res_sample, down_block_additional_residual in zip(
                            down_block_res_samples, down_block_additional_residuals
                        ):
                            down_block_res_sample = (
                                down_block_res_sample + down_block_additional_residual
                            )
                            new_down_block_res_samples += (down_block_res_sample,)

                        down_block_res_samples = new_down_block_res_samples

                    # mid
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        full_mask=full_mask,
                        face_mask=face_mask,
                        lip_mask=lip_mask,
                        audio_embedding=audio_embedding,
                        motion_scale=motion_scale,
                    )

                    if mid_block_additional_residual is not None:
                        sample = sample + mid_block_additional_residual

                    #----------------------save feature-------------------------
                    # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                    setattr(self, 'skip_feature', deepcopy(down_block_res_samples))
                    setattr(self, 'toup_feature', sample.detach().clone())
                    #-----------------------save feature------------------------      

                # else:
                #     down_block_res_samples = self.skip_feature
                #     sample = self.toup_feature

            if mode == 'full' or mode == 'up_only':
                if not order in cond or mode == 'up_only':
                    down_block_res_samples = self.skip_feature
                    sample = self.toup_feature

                #-------------------expand feature for parallel---------------
                if isinstance(timestep, list):
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timestep, sample.shape[0]).to(sample.device)
                    t_emb = self.time_proj(timesteps)

                    # `Timesteps` does not contain any weights and will always return f32 tensors
                    # but time_embedding might actually be running in fp16. so we need to cast here.
                    # there might be better ways to encapsulate this.
                    t_emb = t_emb.to(dtype=self.dtype)

                    emb = self.time_embedding(t_emb)
    
                down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                sample = warpped_feature(sample, step)
                encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                audio_embedding = warpped_audio_tensor(audio_embedding, step)
                full_mask = warpped_face_mask(full_mask, step)
                face_mask = warpped_face_mask(face_mask, step)
                lip_mask = warpped_face_mask(lip_mask, step)

                print('down_block_res_samples[0].shape after warpped_skip_feature:', down_block_res_samples[0].shape)
                print('sample.shape after warpped_feature:', sample.shape)
                print('encoder_hidden_states.shape after warpped_text_emb:', encoder_hidden_states.shape)
                print('')
                #-------------------expand feature for parallel---------------       
                
                print('############################# upblocks ##############################')
                # up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                    down_block_res_samples = down_block_res_samples[
                        : -len(upsample_block.resnets)
                    ]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if (
                        hasattr(upsample_block, "has_cross_attention")
                        and upsample_block.has_cross_attention
                    ):
                        sample, _ = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            full_mask=full_mask,
                            face_mask=face_mask,
                            lip_mask=lip_mask,
                            audio_embedding=audio_embedding,
                            motion_scale=motion_scale,
                            num_parallel=step,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                            encoder_hidden_states=encoder_hidden_states,
                            # audio_embedding=audio_embedding,
                        )

                # print('sample.shape before conv_norm_out:', sample.shape)
                # post-process
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
                sample = self.conv_out(sample)

                if not return_dict:
                    return (sample,)

                return UNet3DConditionOutput(sample=sample)
        return forward
    if model.__class__.__name__ == 'UNet3DConditionModel':
        model.forward = faster_forward(model)