import math

import torch
import xformers
import xformers.ops
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn

from memo.models.attention import zero_module
from memo.models.attention_processor import (
    MemoryLinearAttnProcessor,
)

from memo.models.masking_utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, offset=0):
        x = x + self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x)


class MemoryLinearAttnTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalLinearAttnTransformer(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(
        self,
        hidden_states,
        motion_frames,
        encoder_hidden_states,
        is_new_audio=True,
        update_past_memory=False,
        # bg cache
        mask_fg=None,
        bg_features=None,
        bg_mode=None,
        # save
        block_name='unknown_temporal_module',
        save_dir=None,
        save_layers=None,
        save_types=None,
    ):
        hidden_states = self.temporal_transformer(
            hidden_states,
            motion_frames,
            encoder_hidden_states,
            is_new_audio=is_new_audio,
            update_past_memory=update_past_memory,
            # bg cache
            mask_fg=mask_fg,
            bg_features=bg_features,
            bg_mode=bg_mode,
            # save
            block_name=block_name,
            save_dir=save_dir,
            save_layers=save_layers,
            save_types=save_types,
        )

        output = hidden_states
        return output


class TemporalLinearAttnTransformer(nn.Module):
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
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalLinearAttnTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states,
        motion_frames,
        encoder_hidden_states=None,
        is_new_audio=True,
        update_past_memory=False,
        # save
        block_name='unknown_temporal_transformer_block',
        save_dir=None,
        save_layers=None,
        save_types=None,
        # bg cache
        mask_fg=None,
        bg_features=None,
        bg_mode=None,
    ):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        n_motion_frames = motion_frames.shape[2]

        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        with torch.no_grad():
            motion_frames = rearrange(motion_frames, "b c f h w -> (b f) c h w")

        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        with torch.no_grad():
            motion_frames = self.norm(motion_frames)

        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        with torch.no_grad():
            (
                motion_frames_batch,
                motion_frames_inner_dim,
                motion_frames_height,
                motion_frames_weight,
            ) = motion_frames.shape

            motion_frames = motion_frames.permute(0, 2, 3, 1).reshape(
                motion_frames_batch,
                motion_frames_height * motion_frames_weight,
                motion_frames_inner_dim,
            )
            motion_frames = self.proj_in(motion_frames)

        # Transformer Blocks
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                motion_frames,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                n_motion_frames=n_motion_frames,
                is_new_audio=is_new_audio,
                update_past_memory=update_past_memory,
                # bg cache
                block_name=block_name+f'_temporal_{i}_attn',
                mask_fg=mask_fg,
                bg_features=bg_features,
                bg_mode=bg_mode,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        # print(f'motion_module 204 save_dir: {save_dir}')
        # # module output level
        # if save_dir is not None:    
        #     from .masking_utils import save_attention_data, reshape_attn_data
        #     if 'temporal' in save_types and any(layer in block_name for layer in save_layers):
        #         save_attention_data(save_dir, 'module_output_temporal', reshape_attn_data(output), block_name)

        return output


class TemporalLinearAttnTransformerBlock(nn.Module):
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
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                MemoryLinearAttention(
                    attention_mode=block_name.split("_", maxsplit=1)[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        motion_frames,
        encoder_hidden_states=None,
        video_length=None,
        n_motion_frames=None,
        is_new_audio=True,
        update_past_memory=False,
        # bg cache
        block_name='unknown_temporal_attn_block',
        mask_fg=None,
        bg_features=None,
        bg_mode=None,
    ):
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            with torch.no_grad():
                norm_motion_frames = norm(motion_frames)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    norm_motion_frames,
                    encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                    video_length=video_length,
                    n_motion_frames=n_motion_frames,
                    is_new_audio=is_new_audio,
                    update_past_memory=update_past_memory,
                    # bg cache
                    block_name=block_name,
                    block_idx=i,
                    mask_fg=mask_fg,
                    bg_features=bg_features,
                    bg_mode=bg_mode,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class MemoryLinearAttention(Attention):
    def __init__(
        self,
        *args,
        attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs.get("cross_attention_dim") is not None
        self.query_dim = kwargs["query_dim"]
        self.temporal_position_encoding_max_len = temporal_position_encoding_max_len
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
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op=None,
    ):
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
            processor = MemoryLinearAttnProcessor()
        else:
            processor = MemoryLinearAttnProcessor()

        self.set_processor(processor)

    def forward(
        self,
        hidden_states,
        motion_frames,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        n_motion_frames=None,
        is_new_audio=True,
        update_past_memory=False,
        block_name='unknown_temporal_attn_block',
        block_idx=None,
        mask_fg=None,
        bg_features=None,
        bg_mode=None,
        **cross_attention_kwargs,
    ):
        # print('motion_module 405 hidden_states.shape:', hidden_states.shape, flush=True)
        if bg_mode == 'read':
            # print('motion_module 663 bg_features is not None', flush=True)
            hidden_states = get_fg_tokens(mask_fg, hidden_states)
        # print('motion_module 409 hidden_states.shape:', hidden_states.shape, flush=True)

        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states,
                "(b f) d c -> (b d) f c",
                f=video_length,
            )

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            with torch.no_grad():
                motion_frames = rearrange(motion_frames, "(b f) d c -> (b d) f c", f=n_motion_frames)

            encoder_hidden_states = (
                repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                if encoder_hidden_states is not None
                else encoder_hidden_states
            )

        else:
            raise NotImplementedError

        # print('motion_module 434 hidden_states.shape:', hidden_states.shape, flush=True)
        # print('motion_module 435 motion_frames.shape:', motion_frames.shape, flush=True)
    
        hidden_states = self.processor(
            self,
            hidden_states,
            motion_frames,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            n_motion_frames=n_motion_frames,
            is_new_audio=is_new_audio,
            update_past_memory=update_past_memory,
            **cross_attention_kwargs,
        )

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

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

        return hidden_states
