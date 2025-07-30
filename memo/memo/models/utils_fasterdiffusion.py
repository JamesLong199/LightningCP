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

import time
from typing import Dict, List, Optional, Tuple, Union

# change these for each model
from memo.models.unet_3d import UNet3DConditionOutput, UNet3DConditionModel

from copy import deepcopy

# from memo.pipelines.video_pipeline import VideoPipeline

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

# def warpped_ref_feature(ref_feature_list, step):
#     """
#     Wrapper for expanding reference features for parallel inference.
    
#     Args:
#         ref_feature: Reference feature tensor
    
#     Returns:
#         Expanded reference feature tensor
#     """
#     expanded_ref_feature_list = []
#     for ref_feature in ref_feature_list:
#         expanded_ref_feature = ref_feature[0].repeat(step, 1, 1)
#         expanded_ref_feature_list.append([expanded_ref_feature])
#     return expanded_ref_feature_list

def warpped_ref_feature(ref_features, step):
    """
    Wrapper for expanding reference features for parallel inference.
    
    Args:
        ref_feature: Reference feature tensor
    
    Returns:
        Expanded reference feature tensor
    """
    expanded_ref_features = []
    for i in range(len(ref_features)):
        block_features = []
        if i == 0:
            for j in range(len(ref_features[i])):
                block_features.append(ref_features[i][j].repeat(step, 1, 1, 1))
        else:
            for j in range(len(ref_features[i])):
                block_features.append([ref_features[i][j][0].repeat(step, 1, 1)])
        expanded_ref_features.append(block_features)
    return expanded_ref_features


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
            ref_features: dict,
            timestep: Union[torch.Tensor, float, int, list],
            encoder_hidden_states: torch.Tensor,
            cond: List[int],
            audio_embedding: Optional[torch.Tensor] = None,
            audio_emotion: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.Tensor] = None,
            mask_cond_fea: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            uc_mask: Optional[torch.Tensor] = None,
            # return_dict: bool = True,
            is_new_audio=True,
            update_past_memory=False,
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
            
            if mode == 'full' or mode == 'down_only':
                print("order:", order)
                if order in cond:
                    # pre-process
                    sample = self.conv_in(sample)
                    if mask_cond_fea is not None:
                        sample = sample + mask_cond_fea

                    # down
                    down_block_res_samples = (sample,)
                    for i, downsample_block in enumerate(self.down_blocks):
                        if (
                            hasattr(downsample_block, "has_cross_attention")
                            and downsample_block.has_cross_attention
                        ):
                            sample, res_samples, audio_embedding = downsample_block(
                                hidden_states=sample,
                                ref_feature_list=ref_features["down"][i],
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                                audio_embedding=audio_embedding,
                                emotion=audio_emotion,
                                uc_mask=uc_mask,
                                is_new_audio=is_new_audio,
                                update_past_memory=update_past_memory,
                            )
                            # print("")
                        else:
                            sample, res_samples = downsample_block(
                                hidden_states=sample,
                                ref_feature_list=ref_features["down"][i],
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                is_new_audio=is_new_audio,
                                update_past_memory=update_past_memory,
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
                    sample, audio_embedding = self.mid_block(
                        sample,
                        ref_feature_list=ref_features["mid"][0],
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        audio_embedding=audio_embedding,
                        emotion=audio_emotion,
                        uc_mask=uc_mask,
                        is_new_audio=is_new_audio,
                        update_past_memory=update_past_memory,
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
                # ref_feature_list = 
                sample = warpped_feature(sample, step)
                encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                audio_embedding = warpped_audio_tensor(audio_embedding, step)
                audio_emotion = audio_emotion.repeat(step)
                uc_mask = uc_mask.repeat(step)

                # for i in range(len(self.up_blocks)):
                #     print(f'len(ref_features["up"][{i}]):', len(ref_features["up"][i]))
                #     if not isinstance(ref_features["up"][i][0], list):
                #         print(f'ref_features["up"][{i}][0].shape:', ref_features["up"][i][0].shape)
                #     else:
                #         print('len(ref_features["up"][{i}][0]):', len(ref_features["up"][i][0]))
                #         print(f'ref_features["up"][{i}][0][0].shape:', ref_features["up"][i][0][0].shape)

                ref_features_up = warpped_ref_feature(ref_features["up"], step)


                # print('down_block_res_samples[0].shape after warpped_skip_feature:', down_block_res_samples[0].shape)
                # print('sample.shape after warpped_feature:', sample.shape)
                # print('encoder_hidden_states.shape after warpped_text_emb:', encoder_hidden_states.shape)
              
                #-------------------expand feature for parallel---------------       
                
                print('############################# upblocks ##############################')
                # # up
                # for i in range(len(self.up_blocks)):
                #     print(f'len(ref_features_up[{i}]):', len(ref_features_up[i]))
                #     if not isinstance(ref_features["up"][i][0], list):
                #         print(f'ref_features_up[{i}][0].shape:', ref_features_up[i][0].shape)
                #     else:
                #         print('len(ref_features_up[{i}][0]):', len(ref_features_up[i][0]))
                #         print(f'ref_features_up[{i}][0][0].shape:', ref_features_up[i][0][0].shape)
             

                for i, upsample_block in enumerate(self.up_blocks):
                    # print('upsample_block:', i)
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
                        sample, audio_embedding, _ = upsample_block(
                            hidden_states=sample,
                            # ref_feature_list=warpped_ref_feature(ref_features["up"][i], step),
                            ref_feature_list=ref_features_up[i], 
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            audio_embedding=audio_embedding,
                            emotion=audio_emotion,
                            uc_mask=uc_mask,
                            is_new_audio=is_new_audio,
                            update_past_memory=update_past_memory,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            ref_feature_list=ref_features_up[i], 
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                            encoder_hidden_states=encoder_hidden_states,
                            is_new_audio=is_new_audio,
                            update_past_memory=update_past_memory,
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