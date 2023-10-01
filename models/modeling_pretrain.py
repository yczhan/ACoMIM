# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial

from models.modeling_finetune import Block, _cfg, PatchEmbed, RelativePositionBias
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from .head import *

from diffdist import functional


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
                for _ in range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'beit_small_patch16_224_8k_vocab',
    'beit_base_patch16_224_8k_vocab',
    'beit_large_patch16_224_8k_vocab',
    'moco_deit_small_patch16_224',
    'moco_beit_small_patch16_224',
    'moco_beit_base_patch16_224',
    'beit_base_patch16_224_mlmcls',
]


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, cfg=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos, mask):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        if mask:
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, bool_masked_pos, return_all_tokens=False, mask=True):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos, mask=mask)
        x = x[:, 1:]
        if return_all_tokens:
            x = x.reshape(-1, 768)
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])


class VisionTransformerForMaskedMOCO(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, attn_head_dim=None, use_groupconv=False,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, cfg=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.feat_dim = cfg.model.feat_dim    # the dimension of feature after projection head, used for contrastive learning
        self.momentum = cfg.model.momentum
        self.cos_moment = cfg.model.cos_moment    # whether to use cosine momentum
        self.contrast_temperature = cfg.model.contrast_temperature
        self.queue_len = cfg.model.queue_len
        self.proj_num_layers = cfg.model.proj_num_layers
        assert self.proj_num_layers > 0
        self.pred_num_layers = cfg.model.pred_num_layers
        print('ViT for MaskMOCO  feat_dim:{}, proj_num_layers:{}, pred_num_layers:{}, momentum:{}, temperature:{}, queue_len:{}'\
              .format(self.feat_dim, self.proj_num_layers, self.pred_num_layers, self.momentum, self.contrast_temperature, self.queue_len))
        self.use_cls_feature = cfg.model.use_cls_feature
        print('use_cls_feature :  ', self.use_cls_feature)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_k = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_embed_k = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))   # use different pos_embed to avoid cheating? use group conv?
        else:
            self.pos_embed = self.pos_embed_k = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None
        if use_groupconv:
            print('\n use group conv:', use_groupconv)
            use_abs_pos_emb = False
            use_shared_rel_pos_bias = False
            use_rel_pos_bias = False
        elif use_abs_pos_emb:
            use_shared_rel_pos_bias = False
            use_rel_pos_bias = False
        elif use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)

        print('use_abs_pos_emb:{}, use_rel_pos_bias:{}, use_shared_rel_pos_bias:{}'.format(use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim, use_groupconv=use_groupconv,
            )
            for i in range(depth)])
        self.blocks_k = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim, use_groupconv=use_groupconv,
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        # self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.projector = MLPHead(in_dim=embed_dim, out_dim=self.feat_dim, num_layers=self.proj_num_layers)
        self.projector_k = MLPHead(in_dim=embed_dim, out_dim=self.feat_dim, num_layers=self.proj_num_layers)
        if self.pred_num_layers > 0:
            self.predictor = MLPHead(in_dim=self.feat_dim, out_dim=self.feat_dim, num_layers=self.pred_num_layers)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
            trunc_normal_(self.pos_embed_k, std=self.init_std)  # the initial pos_embed should be the same?
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)

        # trunc_normal_(self.projector.weight, std=self.init_std)
        # trunc_normal_(self.predictor, std=self.init_std)
        # trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)  # apply _init_weights for all submodules, like recursion
        self.fix_init_weight()
        self._copy_weights(self.patch_embed, self.patch_embed_k)
        self._copy_weights(self.blocks, self.blocks_k)
        self._copy_weights(self.projector, self.projector_k)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        if self.pred_num_layers > 0:
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.K = int(1281152 * 1. / dist.get_world_size() / cfg.data.batch_size) * cfg.epochs   # train_data_num
        self.k = int(1281152 * 1. / dist.get_world_size() / cfg.data.batch_size) * cfg.start_epoch

        # create the queue
        self.register_buffer("queue1", torch.randn(self.feat_dim, self.queue_len))
        self.register_buffer("queue2", torch.randn(self.feat_dim, self.queue_len))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _copy_weights(self, m_q, m_k):
        for param_q, param_k in zip(m_q.parameters(), m_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'pos_embed_k', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        _contrast_momentum = 1. - (1. - self.momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2. if self.cos_moment else self.momentum
        self.k = self.k + 1
        for param_q, param_k in zip(self.patch_embed.parameters(), self.patch_embed_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
        for param_q, param_k, in zip(self.blocks.parameters(), self.blocks_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
        for param_q, param_k, in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        keys1 = dist_collect(keys1)
        keys2 = dist_collect(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.T
        self.queue2[:, ptr:ptr + batch_size] = keys2.T
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        N = l_pos.size(0)
        logits = torch.cat([l_pos, l_neg], dim=1)   # logits: Nx(1+K)
        logits /= self.contrast_temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            max_inds = torch.argmax(logits, dim=1)
            corrects = torch.eq(max_inds, 0).float().sum()
            patch_acc = corrects / N
        return loss, patch_acc

    def forward_features(self, x, bool_masked_pos, mask):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        if mask:
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)   # w.shape ([64, 196, 1]), elements are 0 or 1
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        return self.norm(x)

    def forward_features_k(self, x_k, bool_masked_pos):
        with torch.no_grad():
            self._momentum_update_key_encoder()

            x_k = self.patch_embed_k(x_k, bool_masked_pos=bool_masked_pos)
            batch_size, seq_len, _ = x_k.size()

            cls_tokens_k = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x_k = torch.cat((cls_tokens_k, x_k), dim=1)
            if self.pos_embed_k is not None:
                x_k = x_k + self.pos_embed_k
            x_k = self.pos_drop(x_k)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks_k:
                x_k = blk(x_k, rel_pos_bias=rel_pos_bias)
        return self.norm(x_k)

    def forward(self, x, x_k, bool_masked_pos, bool_masked_pos_k, return_all_tokens=False, mask=True):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos, mask=mask)
        x_k = self.forward_features_k(x_k, bool_masked_pos=bool_masked_pos_k)

        if self.use_cls_feature:
            x = x[:, 0]     # torch.Size([64, 768])
            x_k = x_k[:, 0]
        elif return_all_tokens:
            x = x[:, 1:]  # torch.Size([64, 196, 768])  [batch=64, patch_num=14 x 14, patch_embed]
            x_k = x_k[:, 1:]    # elements in x may less than 0
            x = x.mean(1)   # torch.Size([64, 768])
            x_k = x_k.mean(1)
        else:
            '''to calculate the mean value of the masked tokens in an image'''
            # int_masked_pos = bool_masked_pos.to(torch.int)  # [True, False...] -> [1,0...], torch.Size([64, 196])
            # divisor = int_masked_pos.sum(1).expand(int_masked_pos.shape[1], -1).T  # tensor([[73, ...,73, 73], [75,...,75, 75]...,[74,...,74, 74]] ; torch.Size([64, 196])
            # float_masked_pos = int_masked_pos / divisor  # torch.Size([64, 196])
            # float_masked_pos = float_masked_pos.unsqueeze(2).expand(-1, -1, self.embed_dim)  # torch.Size([64, 196, 768]), [0,0,...,0] for unmasked token, [1/75,...,1/75] for masked
            # x = (x * float_masked_pos).sum(1)
            # x_k = (x_k * float_masked_pos).sum(1)  # torch.Size([64, 768])
            x = x[:, 1:]
            x_k = x_k[:, 1:]
            int_masked_pos = bool_masked_pos.to(torch.int)  # [True, False...] -> [1,0...], torch.Size([64, 196])
            int_masked_pos_k = bool_masked_pos_k.to(torch.int)
            x = x.transpose(1, 2)    # torch.Size([64, 768, 196])
            x_k = x_k.transpose(1, 2)
            int_masked_pos = int_masked_pos.unsqueeze(-1).to(torch.float)   # [64, 196, 1]
            int_masked_pos_k = int_masked_pos_k.unsqueeze(-1).to(torch.float)
            x = (x @ int_masked_pos).squeeze(-1)    # torch.Size([64, 768])
            x_k = (x_k @ int_masked_pos_k).squeeze(-1)
            x = x / (int_masked_pos.sum(1))   # [64, 768] / [64, 1]
            x_k = x_k / (int_masked_pos_k.sum(1))

        feat_q = self.projector(x)      # feat_q torch.Size([64, 256])
        if self.pred_num_layers > 0:
            feat_q = self.predictor(feat_q)
        feat_q = F.normalize(feat_q, dim=1)

        with torch.no_grad():
            feat_k = self.projector_k(x_k)
            feat_k = F.normalize(feat_k, dim=1)

        loss, patch_acc = self.contrastive_loss(feat_q, feat_k, self.queue1)
        self._dequeue_and_enqueue(feat_k, feat_k)
        return loss, patch_acc


class VisionTransformerForMLMCLS(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, attn_head_dim=None, num_classes=1000,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.fc_norm = norm_layer(embed_dim)  # use_mean_pooling = False

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

        self.head.weight.data.mul_(0.001)  # init_scale = 0.001
        self.head.bias.data.mul_(0.001)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def get_classifier(self):
        return self.head

    def forward_features(self, x, bool_masked_pos, mask):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        if mask:
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, bool_masked_pos, return_all_tokens=False, mask=True):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos, mask=mask)
        x = x[:, 1:]
        cls_token = x[:, 0]
        if return_all_tokens:
            return self.head(cls_token), self.lm_head(x)
        else:
            # return the masked tokens
            return self.head(cls_token), self.lm_head(x[bool_masked_pos])


@register_model
def beit_small_patch16_224_8k_vocab(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def beit_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def beit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def moco_deit_small_patch16_224(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedMOCO(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def moco_beit_small_patch16_224(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedMOCO(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def moco_beit_base_patch16_224(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedMOCO(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def beit_base_patch16_224_mlmcls(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMLMCLS(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=1000,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
