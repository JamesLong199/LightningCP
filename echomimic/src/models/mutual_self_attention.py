# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

from src.models.attention import TemporalBasicTransformerBlock

from .attention import BasicTransformerBlock

from src.models.masking_utils import *


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        # device=torch.device("cuda"),
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 12 # changed this from 16 to 12
                    + [0] * batch_size * num_images_per_prompt * 12 # changed this from 16 to 12
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            audio_cond_fea: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
            audio_feature_ratio = 3.0,
            block_name='unknown_block',
            mask_fg=None,
            bg_features=None,
            bg_mode=None,
            remove_ref=False,
            # save
            save_dir=None,
            save_layers=None,
            save_types=None,
            num_parallel=1,
        ):
            # print('mutual_self_attention 119 block_name:', block_name, flush=True)
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    bank_feas = [
                        rearrange(
                            d.unsqueeze(1).repeat(1, video_length, 1, 1),
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank
                    ]
                    # modify_norm_hidden_states = torch.cat(
                    #     [norm_hidden_states] + bank_feas, dim=1
                    # )
                    # print('mutual_self_attention 166 num_parallel:', num_parallel, flush=True)
                    # print('mutual_self_attention 167 bank_feas[0].shape:', bank_feas[0].shape, flush=True)
                    # print('mutual_self_attention 170 num_parallel:', num_parallel, flush=True)
                    # print('mutual_self_attention 171 bank_feas[0].shape:', bank_feas[0].shape, flush=True)
                    
                    if num_parallel > 1:
                        # print('mutual_self_attention 169 num_parallel > 1', flush=True)
                        bank_feas = [d.repeat(num_parallel, 1, 1) for d in bank_feas]
                        # print('mutual_self_attention 176 bank_feas[0].shape:', bank_feas[0].shape, flush=True)

                    if bg_mode == 'read':
                        norm_hidden_states = get_fg_tokens(mask_fg, norm_hidden_states)
                        bank_feas = [get_fg_tokens(mask_fg, d) for d in bank_feas]
                        hidden_states = get_fg_tokens(mask_fg, hidden_states)

                    if remove_ref:
                        modify_norm_hidden_states = norm_hidden_states
                    else:
                        # print('bank_feas[0].shape:', bank_feas[0].shape)
                        # print('norm_hidden_states.shape:', norm_hidden_states.shape)
                        # print('mutual_self_attention 184 bank_feas[0].shape:', bank_feas[0].shape, flush=True)
                        # print('mutual_self_attention 185 norm_hidden_states.shape:', norm_hidden_states.shape, flush=True)

                        modify_norm_hidden_states = torch.cat(
                            [norm_hidden_states] + bank_feas, dim=1
                        )
                    # print('block_name:', block_name, flush=True)
                    # print('norm_hidden_states.shape:', norm_hidden_states.shape, flush=True)
                    # print('modify_norm_hidden_states.shape:', modify_norm_hidden_states.shape, flush=True)
                    
                    # Reference Cross-Attention
                    hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            attention_mask=attention_mask,
                            attention_name=block_name+'_attn1',
                            save_dir=save_dir,
                            save_layers=save_layers,
                            save_types=save_types,
                        )
                        + hidden_states
                    )

                    # print('mutual_self_attention 199 hidden_states_uc.shape:', hidden_states_uc.shape, flush=True)
                    if do_classifier_free_guidance:
                        # print('mutual_self_attention 205 num_parallel:', num_parallel, flush=True)
                        # print('mutual_self_attention 206 hidden_states.shape:', hidden_states.shape, flush=True)
                        # print('mutual_self_attention 207 uc_mask.shape:', uc_mask.shape, flush=True)
                        ## inference with cfg
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            if num_parallel > 1:
                                _uc_mask = _uc_mask.repeat(num_parallel)
                                # print('mutual_self_attention 223 _uc_mask', _uc_mask.shape, flush=True)
                            else:
                                _uc_mask = (
                                    torch.Tensor(
                                        [1] * (hidden_states.shape[0] // 2)
                                        + [0] * (hidden_states.shape[0] // 2)
                                    )
                                    .to(device)
                                    .bool()
                                )
                        # print('mutual_self_attention 216 hidden_states_c.device:', hidden_states_c.device, flush=True)
                        # print('mutual_self_attention 216 _uc_mask.device:', _uc_mask.device, flush=True)
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask], # B * 4096 * 768
                                attention_mask=attention_mask,
                                attention_name=block_name+'_attn1',
                                save_dir=save_dir,
                                save_layers=save_layers,
                                save_types=save_types,
                            )
                            + hidden_states[_uc_mask]
                        )
                        # print('mutual_self_attention 247 hidden_states_c[_uc_mask].shape:', hidden_states_c[_uc_mask].shape, flush=True)
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc
                    # print('mutual_self_attention 228 hidden_states.shape:', hidden_states.shape, flush=True)

                    if self.attn2 is not None:
                        # Audio Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        # print('mutual_self_attention 237 norm_hidden_states.shape:', norm_hidden_states.shape, flush=True)
                        # print('mutual_self_attention 238 audio_cond_fea.shape:', audio_cond_fea.shape, flush=True)
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=audio_cond_fea,
                                attention_mask=attention_mask,
                                attention_name=block_name+'_attn2',
                            ) * audio_feature_ratio
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    if bg_mode == 'read':
                        feature_name = block_name[4:]
                        hidden_states = merge_bg_tokens(mask_fg, bg_features, feature_name, hidden_states)
                    elif bg_mode == 'write':
                        feature_name = block_name[4:]
                        bg_features[feature_name] = hidden_states

                    # Temporal-Attention
                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, do_classifier_free_guidance=False, dtype=torch.float16, device=torch.device("cuda:0")):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                if do_classifier_free_guidance:
                    # print('mutual_self_attention 375 w.bank[0].shape:', w.bank[0].shape, flush=True)
                    r.bank = [torch.cat([v, v]).to(device, dtype) for v in w.bank]
                    # print('mutual_self_attention 377 r.bank[0].shape:', r.bank[0].shape, flush=True)
                else:
                    r.bank = [v.clone().to(device, dtype) for v in w.bank]
                # print('mutual_self_attention 378 r.bank[0].shape {} device: {}'.format(r.bank[0].shape, r.bank[0].device), flush=True)



    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()
