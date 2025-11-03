"""Masked time-series modeling for multivariate time-series
transformer (1D analog to masked image modeling) based on
ts_transformer and MAE.

This model is used to pretrain the `mvtst` encoder.

This module is based on methods from the paper:
https://arxiv.org/abs/2111.06377

and modified code from:
https://github.com/facebookresearch/mae
https://github.com/huggingface/pytorch-image-models

"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models import build_model_with_cfg, register_model, PretrainedCfg
from timm.models.vision_transformer import Block
from timm.layers import PatchEmbed1D


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedTimeSeriesModelingTransformer(nn.Module):
    """ Masked time-series modeling with Transformer backbone
    """
    def __init__(self, img_size=5000, patch_size=50, in_chans=8,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True,
                 **kwargs):
        super().__init__()
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed1D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if 1:
            # learnable positional embedding
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches + 1, embed_dim) * .02)
        else:
            # fixed sin-cos embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.Sequential(*[
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if 1:
            # learnable positional embedding
            self.decoder_pos_embed = nn.Parameter(
                torch.randn(1, num_patches + 1, decoder_embed_dim) * .02)
        else:
            # fixed sin-cos embedding
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.Sequential(*[
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * in_chans,
            bias=True,
        ) # decoder to patch

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        if 1:
            nn.init.normal_(self.pos_embed, std=.02)
            nn.init.normal_(self.decoder_pos_embed, std=.02)
        else:
            # initialize (and freeze) pos_embed by sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.patch_embed.num_patches**.5),
                cls_token=True)
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.patch_embed.num_patches**.5),
                cls_token=True)
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively
        # normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, seqs):
        """
        seqs: (B, C, L)
        x: (B, L / patch_size, patch_size * in_chans)
        """
        B, C, L = seqs.shape
        p = self.patch_embed.patch_size
        in_chans = self.in_chans
        assert L % p == 0

        latent_len = L // p
        x = seqs.reshape(shape=(B, in_chans, latent_len, p))
        x = torch.einsum('nchp->nhpc', x)
        x = x.reshape(shape=(B, latent_len, p * in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L / patch_size, patch_size * in_chans)
        seqs: (N, C, L)
        """
        B, LL, CC = x.shape
        p = self.patch_embed.patch_size
        in_chans = self.in_chans
        latent_len = LL
        
        x = x.reshape(shape=(B, latent_len, p, in_chans))
        x = torch.einsum('nhpc->nchp', x)
        seqs = x.reshape(shape=(B, latent_len * p, in_chans))
        return seqs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # prepend cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            # unshuffle
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # prepend cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, seqs, pred, mask):
        """
        seqs: [B, C, L]
        pred: [B, L / p, p * C]
        mask: [B, L / p], 0 is keep, 1 is remove, 
        """
        target = self.patchify(seqs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, seqs, mask_ratio=0.6):
        latent, mask, ids_restore = self.forward_encoder(seqs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L/p, p*C]
        loss = self.forward_loss(seqs, pred, mask)
        return loss, pred, mask


@register_model
def mtsm_base_patch25(**kwargs):
    variant = 'mtsm_base_patch25'
    model_args = dict(patch_size=25, embed_dim=400, depth=12,
                      num_heads=8, decoder_embed_dim=256, decoder_depth=8,
                      decoder_num_heads=16, mlp_ratio=4,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    kwargs['pretrained_cfg'] = PretrainedCfg()
    return build_model_with_cfg(MaskedTimeSeriesModelingTransformer, variant,
                                **dict(model_args, **kwargs))


@register_model
def mtsm_base_patch50(**kwargs):
    variant = 'mtsm_base_patch50'
    model_args = dict(patch_size=50, embed_dim=800, depth=12,
                      num_heads=8, decoder_embed_dim=256, decoder_depth=8,
                      decoder_num_heads=16, mlp_ratio=4,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    kwargs['pretrained_cfg'] = PretrainedCfg()
    return build_model_with_cfg(MaskedTimeSeriesModelingTransformer, variant,
                                **dict(model_args, **kwargs))

    
@register_model
def mtsm_base_patch100(**kwargs):
    variant= 'mtsm_base_patch100'
    model_args = dict( patch_size=100, embed_dim=800, depth=8,
                       num_heads=4, decoder_embed_dim=256, decoder_depth=8,
                       decoder_num_heads=16, mlp_ratio=4,
                       norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    kwargs['pretrained_cfg'] = PretrainedCfg()
    return build_model_with_cfg(MaskedTimeSeriesModelingTransformer, variant,
                                **dict(model_args, **kwargs))

@register_model
def mtsm_large_patch50(**kwargs):
    variant = 'mtsm_large_patch50'
    model_args = dict(patch_size=50, embed_dim=1040, depth=16,
                      num_heads=16, decoder_embed_dim=256, decoder_depth=8,
                      decoder_num_heads=16, mlp_ratio=4,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    kwargs['pretrained_cfg'] = PretrainedCfg()
    return build_model_with_cfg(MaskedTimeSeriesModelingTransformer, variant,
                                **dict(model_args, **kwargs))
