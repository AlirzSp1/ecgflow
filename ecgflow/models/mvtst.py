"""Multivariate time-series transformer based on vision transformer.
"""

from functools import partial

import torch
import torch.nn as nn

from timm import create_model
from timm.models import build_model_with_cfg, register_model, PretrainedCfg, \
    load_state_dict
from timm.models.vision_transformer import VisionTransformer, Block, \
    ParallelScalingBlock
from timm.layers import PatchEmbed1D, DropPath, AttentionPoolLatent, \
    get_norm_layer, SwiGLUPacked

__all__ = []


@register_model
def mvtst_base_patch100(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """Mvtst (Mvtst-B/100) @ 8x5000.
    """
    variant = 'mvtst_base_patch100'
    model_args = dict(patch_size=100, embed_dim=800, depth=8, num_heads=4,
                      mlp_ratio=4, qkv_bias=True, global_pool='avg',
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      embed_layer=PatchEmbed1D)
    input_size = (8, 5000)
    first_conv = 'patch_embed.proj'
    kwargs['pretrained_cfg'] = PretrainedCfg(input_size=input_size,
                                             first_conv=first_conv)
    return build_model_with_cfg(VisionTransformer, variant,
                                pretrained=pretrained,
                                pretrained_strict=False,
                                **dict(model_args, **kwargs))

@register_model
def mvtst_base_patch50(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """MvTsT (MvTsT-B/50) @ 8x5000.
    """
    variant = 'mvtst_base_patch50'
    model_args = dict(patch_size=50, embed_dim=800, depth=12, num_heads=8,
                      mlp_ratio=4, qkv_bias=True, global_pool='avg',
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      embed_layer=PatchEmbed1D,)
    input_size = (8, 5000)
    first_conv = 'patch_embed.proj'
    kwargs['pretrained_cfg'] = PretrainedCfg(input_size=input_size,
                                             first_conv=first_conv)
    return build_model_with_cfg(VisionTransformer, variant,
                                pretrained=pretrained,
                                pretrained_strict=False,
                                **dict(model_args, **kwargs))


@register_model
def mvtst_base_patch25(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """MvTsT (MvTsT-B/25) @ 8x5000.
    """
    variant = 'mvtst_base_patch25'    
    model_args = dict(patch_size=25, embed_dim=800, depth=12, num_heads=8,
                      mlp_ratio=4, qkv_bias=True, global_pool='avg',
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      embed_layer=PatchEmbed1D)
    input_size = (8, 5000)
    first_conv = 'patch_embed.proj'
    kwargs['pretrained_cfg'] = PretrainedCfg(input_size=input_size,
                                             first_conv=first_conv)
    return build_model_with_cfg(VisionTransformer, variant,
                                pretrained=pretrained,
                                pretrained_strict=False,
                                **dict(model_args, **kwargs))


@register_model
def mvtst_large_patch50(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """MvTsT (MvTsT-B/50) @ 8x5000.
    """
    variant = 'mvtst_large_patch50'
    model_args = dict(patch_size=50, embed_dim=1040, depth=16, num_heads=16,
                      mlp_ratio=4, qkv_bias=True, global_pool='avg',
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      embed_layer=PatchEmbed1D, block_fn=ParallelScalingBlock,)
    input_size = (8, 5000)
    first_conv = 'patch_embed.proj'
    kwargs['pretrained_cfg'] = PretrainedCfg(input_size=input_size,
                                             first_conv=first_conv)
    return build_model_with_cfg(VisionTransformer, variant,
                                pretrained=pretrained,
                                pretrained_strict=False,
                                **dict(model_args, **kwargs))

@register_model
def mvtst_tiny_patch1(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """MvTsT (MvTsT-T/1) @ 1x60.
    """
    variant = 'mvtst_tiny_patch1'    
    model_args = dict(patch_size=1, embed_dim=100, depth=4, num_heads=5,
                      mlp_ratio=1, qkv_bias=True, global_pool='avg',
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      embed_layer=PatchEmbed1D)
    input_size = (8, 5000)
    first_conv = 'patch_embed.proj'
    kwargs['pretrained_cfg'] = PretrainedCfg(input_size=input_size,
                                             first_conv=first_conv)
    return build_model_with_cfg(VisionTransformer, variant,
                                pretrained=pretrained,
                                pretrained_strict=False,
                                **dict(model_args, **kwargs))


@register_model
def mvtst_tiny_patch12(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """MvTsT (MvTsT-T/12) @ 1x60.
    """
    variant = 'mvtst_tiny_patch12'
    model_args = dict(patch_size=12, embed_dim=100, depth=4, num_heads=5,
                      mlp_ratio=1, qkv_bias=True, global_pool='avg',
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      embed_layer=PatchEmbed1D,
                      mlp_layer=SwiGLUPacked)
    input_size = (8, 5000)
    first_conv = 'patch_embed.proj'
    kwargs['pretrained_cfg'] = PretrainedCfg(input_size=input_size,
                                             first_conv=first_conv)
    return build_model_with_cfg(VisionTransformer, variant,
                                pretrained=pretrained,
                                pretrained_strict=False,
                                **dict(model_args, **kwargs))
