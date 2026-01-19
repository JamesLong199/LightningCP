# pylint: disable=R0801
# pylint: disable=E1101
# pylint: disable=R0402
# pylint: disable=W1203

"""
This is the main file for the UNet3DConditionModel, which defines the UNet3D model architecture.

The UNet3D model is a 3D convolutional neural network designed for image segmentation and
other computer vision tasks. It consists of an encoder, a decoder, and skip connections between
the corresponding layers of the encoder and decoder. The model can handle 3D data and
performs well on tasks such as image segmentation, object detection, and video analysis.

This file contains the necessary imports, the main UNet3DConditionModel class, and its
methods for setting attention slice, setting gradient checkpointing, setting attention
processor, and the forward method for model inference.

The module provides a comprehensive solution for 3D image segmentation tasks and can be
easily extended for other computer vision tasks as well.
"""

from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME,
                             BaseOutput, logging)
from safetensors.torch import load_file

from .resnet import InflatedConv3d, InflatedGroupNorm
from .unet_3d_blocks import (UNetMidBlock3DCrossAttn, get_down_block,
                             get_up_block)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import time
import math

@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    Data class that serves as the output of the UNet3DConditionModel.

    Attributes:
        sample (`torch.FloatTensor`):
            A tensor representing the processed sample. The shape and nature of this tensor will depend on the 
            specific configuration of the model and the input data.
    """
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    """
    A 3D UNet model designed to handle conditional image and video generation tasks. This model is particularly 
    suited for tasks that require the generation of 3D data, such as volumetric medical imaging or 3D video 
    generation, while incorporating additional conditioning information.

    The model consists of an encoder-decoder structure with skip connections. It utilizes a series of downsampling 
    and upsampling blocks, with a middle block for further processing. Each block can be customized with different 
    types of layers and attention mechanisms.

    Parameters:
        sample_size (`int`, optional): The size of the input sample.
        in_channels (`int`, defaults to 8): The number of input channels.
        out_channels (`int`, defaults to 8): The number of output channels.
        center_input_sample (`bool`, defaults to False): Whether to center the input sample.
        flip_sin_to_cos (`bool`, defaults to True): Whether to flip the sine to cosine in the time embedding.
        freq_shift (`int`, defaults to 0): The frequency shift for the time embedding.
        down_block_types (`Tuple[str]`): A tuple of strings specifying the types of downsampling blocks.
        mid_block_type (`str`): The type of middle block.
        up_block_types (`Tuple[str]`): A tuple of strings specifying the types of upsampling blocks.
        only_cross_attention (`Union[bool, Tuple[bool]]`): Whether to use only cross-attention.
        block_out_channels (`Tuple[int]`): A tuple of integers specifying the output channels for each block.
        layers_per_block (`int`, defaults to 2): The number of layers per block.
        downsample_padding (`int`, defaults to 1): The padding used in downsampling.
        mid_block_scale_factor (`float`, defaults to 1): The scale factor for the middle block.
        act_fn (`str`, defaults to 'silu'): The activation function to be used.
        norm_num_groups (`int`, defaults to 32): The number of groups for normalization.
        norm_eps (`float`, defaults to 1e-5): The epsilon for normalization.
        cross_attention_dim (`int`, defaults to 1280): The dimension for cross-attention.
        attention_head_dim (`Union[int, Tuple[int]]`): The dimension for attention heads.
        dual_cross_attention (`bool`, defaults to False): Whether to use dual cross-attention.
        use_linear_projection (`bool`, defaults to False): Whether to use linear projection.
        class_embed_type (`str`, optional): The type of class embedding.
        num_class_embeds (`int`, optional): The number of class embeddings.
        upcast_attention (`bool`, defaults to False): Whether to upcast attention.
        resnet_time_scale_shift (`str`, defaults to 'default'): The time scale shift for the ResNet.
        use_inflated_groupnorm (`bool`, defaults to False): Whether to use inflated group normalization.
        use_motion_module (`bool`, defaults to False): Whether to use a motion module.
        motion_module_resolutions (`Tuple[int]`): A tuple of resolutions for the motion module.
        motion_module_mid_block (`bool`, defaults to False): Whether to use a motion module in the middle block.
        motion_module_decoder_only (`bool`, defaults to False): Whether to use the motion module only in the decoder.
        motion_module_type (`str`, optional): The type of motion module.
        motion_module_kwargs (`dict`): Keyword arguments for the motion module.
        unet_use_cross_frame_attention (`bool`, optional): Whether to use cross-frame attention in the UNet.
        unet_use_temporal_attention (`bool`, optional): Whether to use temporal attention in the UNet.
        use_audio_module (`bool`, defaults to False): Whether to use an audio module.
        audio_attention_dim (`int`, defaults to 768): The dimension for audio attention.

    The model supports various features such as gradient checkpointing, attention processors, and sliced attention 
    computation, making it flexible and efficient for different computational requirements and use cases.

    The forward method of the model accepts a sample, timestep, and encoder hidden states as input, and it returns 
    the processed sample as output. The method also supports additional conditioning information such as class 
    labels, audio embeddings, and masks for specialized tasks.

    The from_pretrained_2d class method allows loading a pre-trained 2D UNet model and adapting it for 3D tasks by 
    incorporating motion modules and other 3D specific features.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 8,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_inflated_groupnorm=False,
        # Additional
        use_motion_module=False,
        motion_module_resolutions=(1, 2, 4, 8),
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,
        motion_module_kwargs=None,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        # audio
        use_audio_module=False,
        audio_attention_dim=768,
        stack_enable_blocks_name=None,
        stack_enable_blocks_depth=None,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )

        # time
        self.time_proj = Timesteps(
            block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(
                num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [
                only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # print(f"unet_3d 213 attention_head_dim: {attention_head_dim}", flush=True)
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2**i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions)
                and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                use_audio_module=use_audio_module,
                audio_attention_dim=audio_attention_dim,
                depth=i,
                stack_enable_blocks_name=stack_enable_blocks_name,
                stack_enable_blocks_depth=stack_enable_blocks_depth,
            )
            # if down_block_type == 'CrossAttnDownBlock3D':
            #     print(f'unet_3d 254 down_block {i} attentions:', down_block.attentions)
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                use_audio_module=use_audio_module,
                audio_attention_dim=audio_attention_dim,
                depth=3,
                stack_enable_blocks_name=stack_enable_blocks_name,
                stack_enable_blocks_depth=stack_enable_blocks_depth,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                use_audio_module=use_audio_module,
                audio_attention_dim=audio_attention_dim,
                depth=3-i,
                stack_enable_blocks_name=stack_enable_blocks_name,
                stack_enable_blocks_depth=stack_enable_blocks_depth,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                if "temporal_transformer" not in sub_name:
                    fn_recursive_add_processors(
                        f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            if "temporal_transformer" not in name:
                fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = (
            num_slicable_layers * [slice_size]
            if not isinstance(slice_size, list)
            else slice_size
        )

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i, size in enumerate(slice_size):
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(
                    f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(
            module: torch.nn.Module, slice_size: List[int]
        ):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                if "temporal_transformer" not in sub_name:
                    fn_recursive_attn_processor(
                        f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            if "temporal_transformer" not in name:
                fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
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
        # masked attention
        mask_fg: Optional[torch.Tensor] = None,
        use_masked_attention: bool = False,
        masked_layers: Optional[Dict[str, List[str]]] = {'spatial': [], 'audio': [], 'temporal': []},
        remove_ref_list: Optional[List[str]] = [],
        # bg cache
        bg_features: Optional[Dict[str, torch.Tensor]] = None,
        use_cached_bg: bool = False,
        # deepcache
        quick_replicate: bool = False,
        replicate_prv_feature: Optional[List[torch.Tensor]] = None,
        cache_layer_id: Optional[int] = None,
        cache_block_id: Optional[int] = None,
        # superfast cache
        superfast_replicate: bool = False,
        # save
        save_dir: Optional[str] = None,
        save_layers: Optional[List[str]] = None,
        save_types: Optional[List[str]] = None,
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
        # print(f"unet_3d 548 timestep: {timestep}", flush=True) 
        
        timestep_converted = (999-timestep) / 25
        
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
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # print('unet 3d 595 not torch.is_tensor(timesteps)', flush=True)
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                # print('unet 3d 599 isinstance(timestep, float)', flush=True)
                dtype = torch.float32 if is_mps else torch.float64
            else:
                # print('unet 3d 602 not isinstance(timestep, float)', flush=True)
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0: # true
            # print('unet 3d 609 len(timesteps.shape) == 0', flush=True)
            # print('unet 3d 610 timesteps:', timesteps, flush=True) # 999
            timesteps = timesteps[None].to(sample.device)
        
        # assert 0

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # print('unet 3d 605 timesteps:', timesteps, flush=True) # [999]
        timesteps = timesteps.expand(sample.shape[0])
        # print('unet 3d 607 timesteps:', timesteps, flush=True) # [999, 999]

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

        # pre-process
        sample = self.conv_in(sample)
        if mask_cond_fea is not None:
            sample = sample + mask_cond_fea
            # assert 0

        # import numpy as np
        # import os
        # from hallo.models.masking_utils import reshape_attn_data
        # os.makedirs(f'latents_data/t{int(timestep_converted):02d}', exist_ok=True)
        # np.save(f'latents_data/t{int(timestep_converted):02d}/t{int(timestep_converted):02d}_sample.npy', reshape_attn_data(sample).detach().cpu().numpy())

        with open(f'magnitude.txt', 'a') as f:
            f.write(f't{int(timestep_converted):02d}: {torch.linalg.norm(sample)}\n')

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        # print(f"unet_3d 618 mask_fg.keys(): {mask_fg.keys()}", flush=True)

        if mask_fg is None:
            bg_mode = None
        elif bg_features is not None:
            if use_cached_bg:
                bg_mode = 'read'
            elif use_masked_attention:
                bg_mode = 'masked_attention_write'
            else:
                bg_mode = 'write'
        elif use_masked_attention:
            bg_mode = 'masked_attention'
        else:
            raise NotImplementedError()

        # down
        down_block_res_samples = (sample,)
        # print('bg_mode:', bg_mode, flush=True)
        # with open('bg_mode.txt', 'a') as f:
        #     f.write(f'bg_mode: {bg_mode}\n')

        # print('unet 3d 656 quick_replicate:', quick_replicate, flush=True)
        # print('unet 3d 657 replicate_prv_feature is not None:', replicate_prv_feature is not None, flush=True)
        # print('unet 3d 658 len(down_block_res_samples):', len(down_block_res_samples), flush=True)
        if quick_replicate and replicate_prv_feature is not None:

            if not superfast_replicate:
                # for downsample_block in self.down_blocks:
                for i, downsample_block in enumerate(self.down_blocks):
                    if i > cache_layer_id:
                        break

                    if (
                        hasattr(downsample_block, "has_cross_attention")
                        and downsample_block.has_cross_attention
                    ):
                        # start_event.record()
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
                            block_name=f't{int(timestep_converted):02d}_cross_attn_downblock_{i}',
                            mask_fg=mask_fg,
                            bg_features=bg_features,
                            bg_mode=bg_mode,
                            masked_layers=masked_layers,
                            remove_ref_list=remove_ref_list,
                            exist_block_number=cache_block_id if i == cache_layer_id else None,
                        )
                        # print('unet 3d 713 len(res_samples):', len(res_samples), flush=True)
                        # if important_indices_down:
                        #     important_indices.update(important_indices_down)
                        # end_event.record()
                        # end_event.synchronize()
                        # time_elapsed = start_event.elapsed_time(end_event) / 1000
                        # print(f"Downsample block {i} execution time: {time_elapsed} seconds")
                        # print("")
                    else:
                        # start_event.record()
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            # audio_embedding=audio_embedding,
                        )
                        # end_event.record()
                        # end_event.synchronize()
                        # time_elapsed = start_event.elapsed_time(end_event) / 1000
                        # print(f"Downsample block {i} execution time: {time_elapsed} seconds")
                        # print("")

                    down_block_res_samples += res_samples
                    # print('unet 3d 736 len(down_block_res_samples):', len(down_block_res_samples), flush=True)

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

            # print('unet 3d 722 cache_block_id:', cache_block_id, flush=True) # 0
            # print('unet 3d 723 cache_layer_id:', cache_layer_id, flush=True) # 0

            sample = replicate_prv_feature
            if cache_block_id == len(self.down_blocks[cache_layer_id].attentions):
                cache_block_id = 0
                cache_layer_id += 1
            else:
                cache_block_id += 1
            
            # print('unet 3d 731 cache_block_id:', cache_block_id, flush=True) # 1
            # print('unet 3d 732 cache_layer_id:', cache_layer_id, flush=True) # 0
            # assert 0

            # print('unet 3d 751 len(down_block_res_samples):', len(down_block_res_samples), flush=True) # 0
            # up
            for i, upsample_block in enumerate(self.up_blocks):
                
                if i < len(self.up_blocks) - 1 - cache_layer_id: # 4-1-0
                    continue
                if i == len(self.up_blocks) - 1 - cache_layer_id: # 4-1-0
                    trunc_upsample_block = cache_block_id + 1 # 2+1=3
                    # print('unet 3d 759 trunc_upsample_block:', trunc_upsample_block, flush=True)
                else:
                    trunc_upsample_block = len(upsample_block.resnets) 
                    # print('unet 3d 762 trunc_upsample_block:', trunc_upsample_block, flush=True)

                # print('unet 3d 754 i:', i, flush=True)

                is_final_block = i == len(self.up_blocks) - 1 # 4-1=3

                res_samples = down_block_res_samples[-trunc_upsample_block:]
                down_block_res_samples = down_block_res_samples[: -trunc_upsample_block]

        

                # print('unet 3d 769 len(res_samples):', len(res_samples), flush=True) # 1
                # print('unet 3d 770 len(down_block_res_samples):', len(down_block_res_samples), flush=True) # 0
                # print('unet 3d 771 res_samples[0].shape:', res_samples[0].shape, flush=True)
                
                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if (
                    hasattr(upsample_block, "has_cross_attention")
                    and upsample_block.has_cross_attention
                ):
                    # start_event.record()
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
                        block_name=f't{int(timestep_converted):02d}_cross_attn_upblock_{i}',
                        mask_fg=mask_fg,
                        bg_features=bg_features,
                        bg_mode=bg_mode,
                        masked_layers=masked_layers,
                        remove_ref_list=remove_ref_list,
                        enter_block_number=cache_block_id if i == len(self.up_blocks) - 1 - cache_layer_id else None,
                    )
                    # if important_indices_up:
                    #     important_indices.update(important_indices_up)
                    # end_event.record()
                    # end_event.synchronize()
                    # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                    # print(f"Upsample block {i} execution time: {time_elapsed} seconds")
                else:
                    # start_event.record()
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        encoder_hidden_states=encoder_hidden_states,
                        # audio_embedding=audio_embedding,
                    )
                    # end_event.record()
                    # end_event.synchronize()
                    # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                    # print(f"Upsample block {i} execution time: {time_elapsed} seconds")
            
            prv_f = replicate_prv_feature

        else:
            # if mask_fg is None:
            #     bg_mode = None
            # elif use_masked_attention:
            #     if bg_features is None:
            #         bg_mode = 'masked_attention'
            #     else:
            #         bg_mode = 'masked_attention_write'
            # else:
            #     bg_mode = 'write'
            # print('unet_3d 785 bg_mode:', bg_mode, flush=True)
            # for downsample_block in self.down_blocks:
            for i, downsample_block in enumerate(self.down_blocks):
                if (
                    hasattr(downsample_block, "has_cross_attention")
                    and downsample_block.has_cross_attention
                ):  
                    # start_event.record()
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
                        block_name=f't{int(timestep_converted):02d}_cross_attn_downblock_{i}',
                        mask_fg=mask_fg,
                        bg_mode=bg_mode,
                        bg_features=bg_features,
                        masked_layers=masked_layers,
                        remove_ref_list=remove_ref_list,
                        save_dir=save_dir,
                        save_layers=save_layers,
                        save_types=save_types,
                    )

                    # end_event.record()
                    # end_event.synchronize()
                    # time_elapsed = start_event.elapsed_time(end_event) / 1000
                    # print(f"Downsample block {i} execution time: {time_elapsed} seconds")
                    # print("")
                else:
                    # start_event.record()
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        # audio_embedding=audio_embedding,
                    )
                    # end_event.record()
                    # end_event.synchronize()
                    # time_elapsed = start_event.elapsed_time(end_event) / 1000
                    # print(f"Downsample block {i} execution time: {time_elapsed} seconds")
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
            # start_event.record()
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
                block_name=f't{int(timestep_converted):02d}_mid_block',
            )
            # end_event.record()
            # end_event.synchronize()
            # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
            # print(f"Mid block execution time: {time_elapsed} seconds")

            if mid_block_additional_residual is not None:
                sample = sample + mid_block_additional_residual

            if cache_block_id is not None:
                if cache_block_id == len(self.down_blocks[cache_layer_id].attentions) :
                    cache_block_id = 0
                    cache_layer_id += 1
                else:
                    cache_block_id += 1
            prv_f = None
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
                    # start_event.record()
                    sample, current_record_f = upsample_block(
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
                        block_name=f't{int(timestep_converted):02d}_cross_attn_upblock_{i}',
                        mask_fg=mask_fg,
                        bg_features=bg_features,
                        bg_mode=bg_mode,
                        masked_layers=masked_layers,
                        remove_ref_list=remove_ref_list,
                        save_dir=save_dir,
                        save_layers=save_layers,
                        save_types=save_types,
                    )
 
                    # end_event.record()
                    # end_event.synchronize()
                    # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                    # print(f"Upsample block {i} execution time: {time_elapsed} seconds")
                else:
                    # start_event.record()
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        encoder_hidden_states=encoder_hidden_states,
                        # audio_embedding=audio_embedding,
                    )
                    current_record_f = None
                    # end_event.record()
                    # end_event.synchronize()
                    # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                    # print(f"Upsample block {i} execution time: {time_elapsed} seconds")
                
                if cache_layer_id is not None and current_record_f is not None and i == len(self.up_blocks) - cache_layer_id - 1:
                    # print('unet 3d 765 current_record_f is not None:', current_record_f is not None, flush=True)
                    prv_f = current_record_f[-cache_block_id-1]
        

            # for k, v in bg_features.items():
            #     print(f'bg_features {k}: {v.shape}', flush=True)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        # print('unet_3d 894 sample.shape:', sample.shape, flush=True)
        # print('unet_3d 895 prv_f.shape:', prv_f.shape, flush=True)
        if not return_dict:
            return (sample, prv_f)

        return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained_2d(
        cls,
        pretrained_model_path: PathLike,
        motion_module_path: PathLike,
        subfolder=None,
        unet_additional_kwargs=None,
        mm_zero_proj_out=False,
        use_landmark=True,
    ):
        """
        Load a pre-trained 2D UNet model from a given directory.

        Parameters:
            pretrained_model_path (`str` or `PathLike`):
                Path to the directory containing a pre-trained 2D UNet model.
            dtype (`torch.dtype`, *optional*):
                The data type of the loaded model. If not provided, the default data type is used.
            device (`torch.device`, *optional*):
                The device on which the loaded model will be placed. If not provided, the default device is used.
            **kwargs (`Any`):
                Additional keyword arguments passed to the model.

        Returns:
            `UNet3DConditionModel`:
                The loaded 2D UNet model.
        """
        pretrained_model_path = Path(pretrained_model_path)
        motion_module_path = Path(motion_module_path)
        if subfolder is not None:
            pretrained_model_path = pretrained_model_path.joinpath(subfolder)
        logger.info(
            f"loaded temporal unet's pretrained weights from {pretrained_model_path} ..."
        )

        config_file = pretrained_model_path / "config.json"
        if not (config_file.exists() and config_file.is_file()):
            raise RuntimeError(
                f"{config_file} does not exist or is not a file")

        unet_config = cls.load_config(config_file)
        unet_config["_class_name"] = cls.__name__
        unet_config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ]
        unet_config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ]
        unet_config["mid_block_type"] = "UNetMidBlock3DCrossAttn"
        if use_landmark:
            unet_config["in_channels"] = 8
            unet_config["out_channels"] = 8

        model = cls.from_config(unet_config, **unet_additional_kwargs)
        # load the vanilla weights
        if pretrained_model_path.joinpath(SAFETENSORS_WEIGHTS_NAME).exists():
            logger.debug(
                f"loading safeTensors weights from {pretrained_model_path} ..."
            )
            state_dict = load_file(
                pretrained_model_path.joinpath(SAFETENSORS_WEIGHTS_NAME), device="cpu"
            )

        elif pretrained_model_path.joinpath(WEIGHTS_NAME).exists():
            logger.debug(f"loading weights from {pretrained_model_path} ...")
            state_dict = torch.load(
                pretrained_model_path.joinpath(WEIGHTS_NAME),
                map_location="cpu",
                weights_only=True,
            )
        else:
            raise FileNotFoundError(
                f"no weights file found in {pretrained_model_path}")

        # load the motion module weights
        if motion_module_path.exists() and motion_module_path.is_file():
            if motion_module_path.suffix.lower() in [".pth", ".pt", ".ckpt"]:
                print(
                    f"Load motion module params from {motion_module_path}")
                motion_state_dict = torch.load(
                    motion_module_path, map_location="cpu", weights_only=True
                )
            elif motion_module_path.suffix.lower() == ".safetensors":
                motion_state_dict = load_file(motion_module_path, device="cpu")
            else:
                raise RuntimeError(
                    f"unknown file format for motion module weights: {motion_module_path.suffix}"
                )
            if mm_zero_proj_out:
                logger.info(
                    "Zero initialize proj_out layers in motion module...")
                new_motion_state_dict = OrderedDict()
                for k in motion_state_dict:
                    if "proj_out" in k:
                        continue
                    new_motion_state_dict[k] = motion_state_dict[k]
                motion_state_dict = new_motion_state_dict

            # merge the state dicts
            state_dict.update(motion_state_dict)

        model_state_dict = model.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
        # load the weights into the model
        m, u = model.load_state_dict(state_dict, strict=False)
        logger.debug(
            f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        params = [
            p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()
        ]
        logger.info(f"Loaded {sum(params) / 1e6}M-parameter motion module")

        return model
