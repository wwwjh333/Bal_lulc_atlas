# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import torch

from ..common import TwoWayTransformer
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, SamSimMIM


def _checkpoint_to_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _strip_module_prefix(state_dict):
    prefix = "module."
    out = {}
    for k, v in state_dict.items():
        out[k[len(prefix) :] if k.startswith(prefix) else k] = v
    return out


def _matched_state_dict(state_dict, model_sd):
    sd = _strip_module_prefix(dict(state_dict))
    return {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}


def _load_merged_sam_weights(model_sd, sam_official_path, encoder_pretrain_path):
    """Load official sam_ckpt if present, then overwrite image_encoder.* from encoder_pretrain_ckpt."""
    merged = {}
    official = (sam_official_path or "").strip()
    custom = (encoder_pretrain_path or "").strip()
    loaded_official = False

    if official and os.path.isfile(official):
        with open(official, "rb") as f:
            official_sd = _checkpoint_to_state_dict(torch.load(f, map_location="cpu"))
        merged.update(_matched_state_dict(official_sd, model_sd))
        loaded_official = True

    if custom and os.path.isfile(custom):
        with open(custom, "rb") as f:
            custom_sd = _checkpoint_to_state_dict(torch.load(f, map_location="cpu"))
        custom_matched = _matched_state_dict(custom_sd, model_sd)
        if loaded_official:
            enc_prefix = "image_encoder."
            for k, v in custom_matched.items():
                if k.startswith(enc_prefix):
                    merged[k] = v
        else:
            merged = custom_matched

    return merged


def build_sam_vit_h(args=None):
    return _build_sam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
    )


def build_sam_vit_l(args):
    return _build_sam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
    )


def build_sam_vit_b(args):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    )


sam_model_registry = {
    "default": build_sam_vit_b,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
):
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    if args.model_type == "simmim":
        sam = SamSimMIM(
            image_encoder=ImageEncoderViT(
                args = args,
                in_chans=3,
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            embed_dim=encoder_embed_dim,
            feat_dim=prompt_embed_dim,
            patch_size=vit_patch_size,
            in_chans=4,
        )
    else:
        sam = Sam(
            args,
            image_encoder=ImageEncoderViT(
                args = args,
                in_chans=3,
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                # use_rel_pos=False,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=args.num_classes,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        )

    model_sd = sam.state_dict()

    if isinstance(sam, Sam):
        official = (getattr(args, "sam_ckpt", "") or "").strip()
        custom = (getattr(args, "encoder_pretrain_ckpt", "") or "").strip()
        have_official = bool(official) and os.path.isfile(official)
        have_custom = bool(custom) and os.path.isfile(custom)
        if have_official or have_custom:
            merged = _load_merged_sam_weights(
                model_sd,
                official if have_official else "",
                custom if have_custom else "",
            )
            if merged:
                sam.load_state_dict(merged, strict=False)
                o_msg = official if have_official else "(not loaded)"
                c_msg = custom if have_custom else "(not loaded)"
                print(
                    f"[build_sam] merged load: {len(merged)}/{len(model_sd)} keys "
                    f"(sam_ckpt={o_msg}, encoder_pretrain_ckpt={c_msg})"
                )
    else:
        ckpt_path = (getattr(args, "sam_ckpt", "") or "").strip()
        if ckpt_path and os.path.isfile(ckpt_path):
            with open(ckpt_path, "rb") as f:
                ckpt = torch.load(f, map_location="cpu")
                state_dict = _checkpoint_to_state_dict(ckpt)

            matched = _matched_state_dict(state_dict, model_sd)
            sam.load_state_dict(matched, strict=False)
            print(f"[build_sam] loaded {len(matched)}/{len(model_sd)} keys from sam_ckpt={ckpt_path}")

    return sam
