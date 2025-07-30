# pylint: disable=R0801
# pylint: disable=W0613
# pylint: disable=W0221

"""
temporal_transformers.py

This module provides classes and functions for implementing Temporal Transformers
in PyTorch, designed for handling video data and temporal sequences within transformer-based models.

Functions:
    zero_module(module)
        Zero out the parameters of a module and return it.

Classes:
    TemporalTransformer3DModelOutput(BaseOutput)
        Dataclass for storing the output of TemporalTransformer3DModel.

    VanillaTemporalModule(nn.Module)
        A Vanilla Temporal Module class for handling temporal data.

    TemporalTransformer3DModel(nn.Module)
        A Temporal Transformer 3D Model class for transforming temporal data.

    TemporalTransformerBlock(nn.Module)
        A Temporal Transformer Block class for building the transformer architecture.

    PositionalEncoding(nn.Module)
        A Positional Encoding module for transformers to encode positional information.

Dependencies:
    math
    dataclasses.dataclass
    typing (Callable, Optional)
    torch
    diffusers (FeedForward, Attention, AttnProcessor)
    diffusers.utils (BaseOutput)
    diffusers.utils.import_utils (is_xformers_available)
    einops (rearrange, repeat)
    torch.nn
    xformers
    xformers.ops

Example Usage:
    >>> motion_module = get_motion_module(in_channels=512, motion_module_type="Vanilla", motion_module_kwargs={})
    >>> output = motion_module(input_tensor, temb, encoder_hidden_states)

This module is designed to facilitate the creation, training, and inference of transformer models
that operate on temporal data, such as videos or time-series. It includes mechanisms for applying temporal attention,
managing positional encoding, and integrating with external libraries for efficient attention operations.
"""

# This code is copied from https://github.com/guoyww/AnimateDiff.

import math

import torch
import xformers
import xformers.ops
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn
from .masking_utils import *

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    
    Args:
    - module: A PyTorch module to zero out its parameters.

    Returns:
    A zeroed out PyTorch module.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporalTransformer3DModelOutput(BaseOutput):
    """
    Output class for the TemporalTransformer3DModel.
    
    Attributes:
        sample (torch.FloatTensor): The output sample tensor from the model.
    """
    sample: torch.FloatTensor

    def get_sample_shape(self):
        """
        Returns the shape of the sample tensor.
        
        Returns:
        Tuple: The shape of the sample tensor.
        """
        return self.sample.shape


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    """
    This function returns a motion module based on the given type and parameters.
    
    Args:
    - in_channels (int): The number of input channels for the motion module.
    - motion_module_type (str): The type of motion module to create. Currently, only "Vanilla" is supported.
    - motion_module_kwargs (dict): Additional keyword arguments to pass to the motion module constructor.
    
    Returns:
    VanillaTemporalModule: The created motion module.
    
    Raises:
    ValueError: If an unsupported motion_module_type is provided.
    """
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )

    raise ValueError


class VanillaTemporalModule(nn.Module):
    """
    A Vanilla Temporal Module class.

    Args:
    - in_channels (int): The number of input channels for the motion module.
    - num_attention_heads (int): Number of attention heads.
    - num_transformer_block (int): Number of transformer blocks.
    - attention_block_types (tuple): Types of attention blocks.
    - cross_frame_attention_mode: Mode for cross-frame attention.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    - temporal_attention_dim_div (int): Divisor for temporal attention dimension.
    - zero_initialize (bool): Flag for zero initialization.
    """

    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def forward(
        self,
        input_tensor,
        encoder_hidden_states,
        attention_mask=None,
        block_name='unknown_temporal_transformer_block',
        mask_fg=None,
        bg_features=None,
        bg_mode=None,
        # save
        save_dir=None,
        save_layers=None,
        save_types=None,
    ):
        """
        Forward pass of the TemporalTransformer3DModel.

        Args:
            hidden_states (torch.Tensor): The hidden states of the model.
            encoder_hidden_states (torch.Tensor, optional): The hidden states of the encoder.
            attention_mask (torch.Tensor, optional): The attention mask.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states,
            block_name=block_name,
            mask_fg=mask_fg,
            bg_features=bg_features,
            bg_mode=bg_mode,
            # save
            save_dir=save_dir,
            save_layers=save_layers,
            save_types=save_types,
        )

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    """
    A Temporal Transformer 3D Model class.

    Args:
    - in_channels (int): The number of input channels.
    - num_attention_heads (int): Number of attention heads.
    - attention_head_dim (int): Dimension of attention heads.
    - num_layers (int): Number of transformer layers.
    - attention_block_types (tuple): Types of attention blocks.
    - dropout (float): Dropout rate.
    - norm_num_groups (int): Number of groups for normalization.
    - cross_attention_dim (int): Dimension for cross-attention.
    - activation_fn (str): Activation function.
    - attention_bias (bool): Flag for attention bias.
    - upcast_attention (bool): Flag for upcast attention.
    - cross_frame_attention_mode: Mode for cross-frame attention.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    """
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, block_name='unknown_temporal_transformer_block', 
                mask_fg=None,
                bg_features=None,
                bg_mode=None,
                # save
                save_dir=None,
                save_layers=None,
                save_types=None,
    ):
        """
        Forward pass for the TemporalTransformer3DModel.

        Args:
            hidden_states (torch.Tensor): The input hidden states with shape (batch_size, sequence_length, in_channels).
            encoder_hidden_states (torch.Tensor, optional): The encoder hidden states with shape (batch_size, encoder_sequence_length, in_channels).

        Returns:
            torch.Tensor: The output hidden states with shape (batch_size, sequence_length, in_channels).
        """
        # print('motion_module 286 block_name:', block_name, flush=True)
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        # print(f"motion_module 285 forward hidden_states.shape: {hidden_states.shape}", flush=True) # 2, 320, 18, 64, 64
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # print(f"motion_module 287 forward hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 320, 64, 64
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        # print('motion_module 296 hidden_states.shape:', hidden_states.shape, flush=True) # 36, 320, 64, 64
        hidden_states = self.norm(hidden_states)
        # print('motion_module 298 hidden_states.shape:', hidden_states.shape, flush=True) # 36, 320, 64, 64
        # end_event.record()
        # end_event.synchronize()
        # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
        # print(f'motion_module 300 norm time: {time_elapsed}', flush=True)

        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim
        )
        # print(f"motion_module 296 forward hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 4096, 320
        hidden_states = self.proj_in(hidden_states)
        # print(f"motion_module 298 forward hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 4096, 320

        # Transformer Blocks
        # print('number of transformer blocks:', len(self.transformer_blocks))
        for i, block in enumerate(self.transformer_blocks):
            # start_event = torch.cuda.Event(enable_timing=True)  
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                block_name=block_name+f'_temporal_{i}_attn',
                mask_fg=mask_fg,
                bg_features=bg_features,
                bg_mode=bg_mode,
                # save
                save_dir=save_dir,
                save_layers=save_layers,
                save_types=save_types,
            )
            # end_event.record()
            # end_event.synchronize()
            # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
            # print(f'motion_module 325 block {i} time: {time_elapsed}', flush=True)
        # print(f"motion_module 307 forward hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 4096, 320
        # output    
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        hidden_states = self.proj_out(hidden_states)
        # end_event.record()
        # end_event.synchronize()
        # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
        # print(f'motion_module 335 proj_out time: {time_elapsed}', flush=True)
        # print(f"motion_module 310 forward hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 4096, 320
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # print(f"motion_module 313 forward hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 320, 64, 64
        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        # print(f"motion_module 319 forward output.shape: {output.shape}", flush=True) # 2, 320, 18, 64, 64

        # # module output level
        # if save_dir is not None:    
        #     if 'temporal' in save_types and any(layer in block_name for layer in save_layers):
        #         save_attention_data(save_dir, 'module_output_temporal', reshape_attn_data(output), block_name)

        return output


class TemporalTransformerBlock(nn.Module):
    """
    A Temporal Transformer Block class.

    Args:
    - dim (int): Dimension of the block.
    - num_attention_heads (int): Number of attention heads.
    - attention_head_dim (int): Dimension of attention heads.
    - attention_block_types (tuple): Types of attention blocks.
    - dropout (float): Dropout rate.
    - cross_attention_dim (int): Dimension for cross-attention.
    - activation_fn (str): Activation function.
    - attention_bias (bool): Flag for attention bias.
    - upcast_attention (bool): Flag for upcast attention.
    - cross_frame_attention_mode: Mode for cross-frame attention.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for i, block_name in enumerate(attention_block_types):
            # print('motion_module 402 block_name:', block_name, flush=True)
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_", maxsplit=1)[0],
                    cross_attention_dim=cross_attention_dim
                    if block_name.endswith("_Cross")
                    else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))
        # assert 0

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        video_length=None,
        block_name='unknown_temporal_attn_block',
        mask_fg=None,
        bg_features=None,
        bg_mode=None,
        # save
        save_dir=None,
        save_layers=None,
        save_types=None,
    ):
        """
        Forward pass for the TemporalTransformerBlock.

        Args:
            hidden_states (torch.Tensor): The input hidden states with shape
                (batch_size, video_length, in_channels).
            encoder_hidden_states (torch.Tensor, optional): The encoder hidden states
                with shape (batch_size, encoder_length, in_channels).
            video_length (int, optional): The length of the video.

        Returns:
            torch.Tensor: The output hidden states with shape
                (batch_size, video_length, in_channels).
        """
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            
            # start_event.record()
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
                    block_name=block_name,
                    block_idx=i,
                    mask_fg=mask_fg,
                    bg_features=bg_features,
                    bg_mode=bg_mode,
                    # save
                    save_dir=save_dir,
                    save_layers=save_layers,
                    save_types=save_types,
                )
                + hidden_states
            )
            # end_event.record()
            # end_event.synchronize()
            # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
            # print(f"temporal attn {i} took {time_elapsed} seconds", flush=True)
        
        # 0417
        # if save_dir is not None:    
        #     if 'temporal' in save_types and any(layer in block_name for layer in save_layers):
        #         save_attention_data(save_dir, 'attn', hidden_states, block_name)

        # start_event.record()
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        # end_event.record()
        # end_event.synchronize()
        # time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
        # print(f"temporal ff took {time_elapsed} seconds", flush=True)

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for transformers.

    Args:
    - d_model (int): Model dimension.
    - dropout (float): Dropout rate.
    - max_len (int): Maximum length for positional encoding.
    """
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        This method takes an input tensor `x` and adds the positional encoding to it. The positional encoding is
        generated based on the input tensor's shape and is added to the input tensor element-wise.

        Args:
            x (torch.Tensor): The input tensor to be positionally encoded.

        Returns:
            torch.Tensor: The positionally encoded tensor.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(Attention):
    """
    Versatile Attention class.

    Args:
    - attention_mode: Attention mode.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    """
    def __init__(
        self,
        *args,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs.get("cross_attention_dim") is not None
        # print('motion_module 447 VersatileAttention.is_cross_attention:', self.is_cross_attention, flush=True)
        # assert 0

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        """
        Returns a string representation of the module with information about the attention mode and whether it is cross-attention.
        
        Returns:
            str: A string representation of the module.
        """
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op = None,
    ):
        """
        Sets the use of memory-efficient attention xformers for the VersatileAttention class.

        Args:
            use_memory_efficient_attention_xformers (bool): A boolean flag indicating whether to use memory-efficient attention xformers or not.

        Returns:
            None

        """
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )

            if not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )

            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            processor = AttnProcessor()
        else:
            processor = AttnProcessor()

        self.set_processor(processor)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        block_name='unknown_temporal_attn_block',
        block_idx=None,
        mask_fg=None,
        bg_features=None,
        bg_mode=None,
        # save
        save_dir=None,
        save_layers=None,
        save_types=None,
        **cross_attention_kwargs,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                The hidden states to be passed through the model.
            encoder_hidden_states (`torch.Tensor`, optional):
                The encoder hidden states to be passed through the model.
            attention_mask (`torch.Tensor`, optional):
                The attention mask to be used in the model.
            video_length (`int`, optional):
                The length of the video.
            cross_attention_kwargs (`dict`, optional):
                Additional keyword arguments to be used for cross-attention.

        Returns:
            `torch.Tensor`:
                The output tensor after passing through the model.

        """
        
        if bg_mode == 'read':
            # print('motion_module 663 bg_features is not None', flush=True)
            hidden_states = get_fg_tokens(mask_fg, hidden_states)

        # print('motion_module 662 hidden_states.shape:', hidden_states.shape, flush=True)
        # print('motion_module 663 encoder_hidden_states.shape:', encoder_hidden_states.shape, flush=True)

        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]  # d means HxW
            # print(f"versatile_attention 587 hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 4096, 320
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            # print(f"versatile_attention 593 hidden_states.shape: {hidden_states.shape}", flush=True) # 8192, 18, 320
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)
            # print(f"versatile_attention 598 encoder_hidden_states: {encoder_hidden_states}", flush=True) # None
            encoder_hidden_states = (
                repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                if encoder_hidden_states is not None
                else encoder_hidden_states
            )

        else:
            raise NotImplementedError

        # print('motion_module 692 hidden_states.shape:', hidden_states.shape, flush=True)
        # print('motion_module 693 encoder_hidden_states.shape:', encoder_hidden_states.shape, flush=True)


        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            # attention_name=block_name,
            **cross_attention_kwargs,
        )

        # if (mask_fg is None or int(math.sqrt(hidden_states.shape[1])) not in mask_fg)  \
        # or ('downblock_0_layer_0' not in block_name and 'upblock_3_layer_0' not in block_name and 'upblock_3_layer_1' not in block_name and 'upblock_3_layer_2' not in block_name):
               
        #     hidden_states = self.processor(
        #         self,
        #         hidden_states,
        #         encoder_hidden_states=encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         # attention_name=block_name,
        #         **cross_attention_kwargs,
        #     )

        # else:

        #     hidden_states_fg = get_masked_tokens(mask_fg, hidden_states, 'fg')
        #     hidden_states_bg = get_masked_tokens(mask_fg, hidden_states, 'bg')

        #     hidden_states_fg = self.processor(
        #         self,
        #         hidden_states_fg,
        #         encoder_hidden_states=encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         # attention_name=block_name,
        #         **cross_attention_kwargs,
        #     )

        #     hidden_states_bg = self.processor(
        #         self,
        #         hidden_states_bg,
        #         encoder_hidden_states=encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         # attention_name=block_name,
        #         **cross_attention_kwargs,
        #     )

        #     hidden_states = merge_bgfg_features(mask_fg, hidden_states_bg, hidden_states_fg)

        # print('motion_module 742 hidden_states.shape:', hidden_states.shape, flush=True)
        if self.attention_mode == "Temporal":
            hidden_states = rearrange(
                hidden_states, "(b d) f c -> (b f) d c", d=d)
        # print(f"versatile_attention 620 hidden_states.shape: {hidden_states.shape}", flush=True) # 36, 4096, 320

        # bg_cache level
        if save_dir is not None:  
            if 'temporal' in save_types and any(layer in block_name for layer in save_layers):
                save_attention_data(save_dir, f'bg_cache_{block_idx}', hidden_states, block_name)   

        if bg_mode == 'read':
            # print('motion_module 749 hidden_states.shape:', hidden_states.shape, flush=True)
            feature_name = block_name[4:] + f'_idx_{block_idx}'
            # print('motion_module 749 bg_features.keys():', bg_features.keys(), flush=True)
            # print('motion_module 750 feature_name:', feature_name, flush=True)
            hidden_states = merge_bg_tokens(mask_fg, bg_features, feature_name, hidden_states)
        elif bg_mode == 'write' or bg_mode == 'masked_attention_write':
            # print('motion_module 753 hidden_states.shape:', hidden_states.shape, flush=True)
            feature_name = block_name[4:] + f'_idx_{block_idx}'
            bg_features[feature_name] = hidden_states
        # elif bg_mode == 'masked_attention_write' or bg_mode == 'masked_attention':
        #     raise NotImplementedError
        # print('motion_module 758 hidden_states.shape:', hidden_states.shape, flush=True)
        return hidden_states
