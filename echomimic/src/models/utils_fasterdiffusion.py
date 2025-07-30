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

from src.models.mutual_self_attention import ReferenceAttentionControl
import time
from typing import Dict, List, Optional, Tuple, Union
# from hallo.models.unet_3d_original import UNet3DConditionOutput, UNet3DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.unet_3d_echo import UNet3DConditionOutput

from copy import deepcopy

from src.pipelines.pipeline_fasterdiffusion import Audio2VideoPipeline

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
    #     if semi_bs >= 1:
    #         texp = timestep.expand(semi_bs)
    #     else:
    #         texp = timestep
    #     ts.append(texp)
    # timesteps = torch.cat(ts)
    # print('126 timesteps:', timesteps, flush=True)
    # print('127 bs:', bs, flush=True)
    # print('128 len(timesteps):', len(timesteps), flush=True)

    semi_bs = bs // 2
    ts = []
    for timestep in timesteps:
        timestep = timestep[None]
        if bs == 2: # original
            texp = timestep.expand(semi_bs)
            ts.append(texp)
            ts.append(texp)
        elif bs == 1: # acc
            texp = timestep
            ts.append(texp)
        else:
            raise ValueError(f"Invalid batch size: {bs}")
    timesteps = torch.cat(ts)
    return timesteps.reshape(-1)  


def register_time(model, time):
    """
    Register current timestep in model.
    
    Args:
        model: UNet model
        time: Current timestep
    """
    setattr(model, 'order', time)


def register_faster_forward_multi_gpu(model, mod = '50ls'):
    def faster_forward(self):
        def forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            cond: List[int],
            class_labels: Optional[torch.Tensor] = None,
            audio_cond_fea: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            face_musk_fea: Optional[torch.Tensor] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            # faster diffusion
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

            # print('utils_fasterdiffusion 237 timesteps:', timesteps, flush=True)
            # faster diffusion
            if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0])
                # print('utils_fasterdiffusion 242 timesteps:', timesteps, flush=True)
            elif isinstance(timesteps, list):
                #timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)
                # print('utils_fasterdiffusion 246 timesteps:', timesteps, flush=True)
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
                # print("order:", order)
                if order in cond:
                    # pre-process
                    sample = self.conv_in(sample)
                    if face_musk_fea is not None:
                        sample = sample + face_musk_fea

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
                                audio_cond_fea=audio_cond_fea,
                                attention_mask=attention_mask,
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
                        audio_cond_fea=audio_cond_fea,
                        attention_mask=attention_mask,
                    )

                    if mid_block_additional_residual is not None:
                        sample = sample + mid_block_additional_residual

                    #----------------------save feature-------------------------
                    # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                    setattr(self, 'skip_feature', deepcopy(down_block_res_samples))
                    setattr(self, 'toup_feature', sample.detach().clone())

                    # print('utils_fasterdiffusion 336 sample.shape:', sample.shape, flush=True)
                    # for i in range(len(down_block_res_samples)):
                    #     print('utils_fasterdiffusion 337 down_block_res_samples[{}].shape:'.format(i), down_block_res_samples[i].shape, flush=True)
                    # # assert 0
                    #-----------------------save feature------------------------      

                # else:
                #     down_block_res_samples = self.skip_feature
                #     sample = self.toup_feature

            if mode == 'full' or mode == 'up_only':
                if not order in cond or mode == 'up_only':
                    down_block_res_samples = self.skip_feature
                    sample = self.toup_feature

                # print('timestep:', timestep, flush=True)
                #-------------------expand feature for parallel---------------
                if isinstance(timestep, list):
                    #timesteps list, such as [981,961,941]
                    # print('utils_fasterdiffusion 367 timestep:', timestep, flush=True)
                    # print('utils_fasterdiffusion 368 sample.shape:', sample.shape, flush=True)
                    timesteps = warpped_timestep(timestep, sample.shape[0]).to(sample.device)
                    # print('utils_fasterdiffusion 369 timesteps:', timesteps, flush=True)
                    t_emb = self.time_proj(timesteps)

                    # `Timesteps` does not contain any weights and will always return f32 tensors
                    # but time_embedding might actually be running in fp16. so we need to cast here.
                    # there might be better ways to encapsulate this.
                    t_emb = t_emb.to(dtype=self.dtype)

                    emb = self.time_embedding(t_emb)

                # print('utils_fasterdiffusion 380 down_block_res_samples[0].shape before warpped_skip_feature:', down_block_res_samples[0].shape)
                # print('utils_fasterdiffusion 381 sample.shape before warpped_feature:', sample.shape)
                # print('utils_fasterdiffusion 382 audio_cond_fea.shape before warpped_audio_tensor:', audio_cond_fea.shape)
                

                down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                sample = warpped_feature(sample, step)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                audio_cond_fea = warpped_audio_tensor(audio_cond_fea, step)
               

                # print('utils_fasterdiffusion 387 down_block_res_samples[0].shape after warpped_skip_feature:', down_block_res_samples[0].shape)
                # print('utils_fasterdiffusion 388 sample.shape after warpped_feature:', sample.shape)
                # print('utils_fasterdiffusion 389 audio_cond_fea.shape after warpped_audio_tensor:', audio_cond_fea.shape)
                
                #-------------------expand feature for parallel---------------       
                
                # print('############################# upblocks ##############################')
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
                            audio_cond_fea=audio_cond_fea,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            num_parallel=step,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                            encoder_hidden_states=encoder_hidden_states,
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
    if model.__class__.__name__ == 'EchoUNet3DConditionModel':
        model.forward = faster_forward(model)