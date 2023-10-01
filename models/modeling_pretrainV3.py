'''
pretraining model V3, symmetric on V2
'''


import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial, reduce
from operator import mul

from models.modeling_finetune import _cfg, PatchEmbed, RelativePositionBias
from models.pos_embed import get_2d_sincos_pos_embed
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


def generate_maskmap_from_attn(attentions, mask_ratio=0.4, mask_shape='random'):
    '''
    :param attentions: attention map from the ViT block
    :param mask_ratio: mask percentage
    :return: tensor bool mask map, (B, W*H)
    '''
    w_featmap = h_featmap = int((attentions.shape[2] - 1) ** 0.5)    # attentions : tensor [B, num_head, 197, 197]
    bs = attentions.shape[0]  # batch size
    nh = attentions.shape[1]  # number of head
    # we keep only the output patch attention
    attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)    # [B, num_head, 196]
    attn_avg = attentions.mean(1)   # [B, 196]

    if mask_shape == 'random':  # select the patches that have highest score
        val, idx = torch.sort(attn_avg, descending=True)
        mask_num = int(mask_ratio * attn_avg.shape[-1])
        mask_map = (attn_avg - val[..., mask_num].unsqueeze(-1)) > 0   # negative -> False, positive -> True
    elif mask_shape == 'box':
        min_num = int(attn_avg.shape[-1] * (mask_ratio - 0.1))  # minimum mask token
        max_num = int(attn_avg.shape[-1] * (mask_ratio + 0.1))
        min_len = math.ceil(math.ceil(min_num / 2) ** 0.5)  # assert that the box ratio is 0.5 ~ 2
        max_len = int((max_num * 2) ** 0.5)  # min & max length of box side
        importance_np = attn_avg
        min_value, _ = importance_np.min(axis=-1)   # value, index
        max_value, _ = importance_np.max(axis=-1)
        importance_np = importance_np.reshape(bs, h_featmap, w_featmap)
        min_value = min_value.unsqueeze(-1).unsqueeze(-1)   # [bs, 1, 1] in order to match the dimension of attention map
        max_value = max_value.unsqueeze(-1).unsqueeze(-1)

        importance_np = (importance_np - min_value) / (max_value - min_value + 1e-6)
        integral = torch.cumsum(torch.cumsum(importance_np, axis=-1), axis=-2).cpu().numpy()  # [B, H, W]

        # generate the height and width of mask box for each sample in batch
        h = torch.zeros(bs)
        w = torch.zeros(bs)
        for i in range(bs):
            for attempt in range(10):
                h[i] = random.randint(min_len, max_len)
                w[i] = random.randint(min_len, max_len)
                if 2 * w[i] > h[i] > 0.5 * w[i] and max_num > h[i] * w[i] > min_num and w[i] < w_featmap and h[i] < h_featmap:
                    break
                elif attempt == 9:
                    h[i] = w[i] = math.ceil(((min_num + max_num) / 2) ** 0.5)

        mask_map = torch.zeros(size=(bs, w_featmap, h_featmap), dtype=bool)
        for idx in range(bs):
            # propose dozens of bounding boxes, 20 boxes by default
            h_i = int(h[idx])   # the height of mask box in sample idx
            w_i = int(w[idx])
            y1 = torch.randint(0, h_featmap - h_i + 1, size=(20,)).numpy()  # side_len[1] - h + 1 is exclusive
            x1 = torch.randint(0, w_featmap - w_i + 1, size=(20,)).numpy()
            y2 = y1 + h_i - 1
            x2 = x1 + w_i - 1

            sum1 = integral[idx, y1, x1]
            sum2 = integral[idx, y1, x2]
            sum3 = integral[idx, y2, x2]
            sum4 = integral[idx, y2, x1]
            scores = sum3 + sum1 - sum2 - sum4

            sorted_inds = np.argsort(scores)[::-1]
            keep_lens = max(int(len(sorted_inds) * 0.1), 1)
            select_idx = int(sorted_inds[random.randint(0, keep_lens - 1)])
            i, j = int(y1[select_idx]), int(x1[select_idx])

            for box_w in range(w_i):
                for box_h in range(h_i):
                    mask_map[idx, i + box_h, j + box_w] = True

        mask_map = mask_map.reshape(bs, -1).cuda()

    return mask_map


def generate_intra_neg_map(bool_mask_map, intra_num=15, type='box'):
    '''
    Args:
        bool_mask_map: torch bool [B, H*W]
        intra_num: the number of intra negatives per sample (K)
        type: the mask map type, box by default
    Returns: mask map: torch bool [B, K, H*W]
    '''
    B, C = bool_mask_map.shape[0], bool_mask_map.shape[-1]
    H = W = int(C**0.5)  # 196 ** 0.5 = 14
    bool_mask_map = ~bool_mask_map  # True -> False, False -> True
    int_mask_map = bool_mask_map.to(torch.int).reshape(B, H, W)
    integral = torch.cumsum(torch.cumsum(int_mask_map, axis=-1), axis=-2).cpu().numpy()  # [B, H, W]
    intra_mask_map = int_mask_map.unsqueeze(1).repeat(1, intra_num, 1, 1)   # [B, K, H ,W]

    min_num = 16    # the minimum mask token number
    min_len = 5
    max_len = 9
    for idx in range(B):
        for k in range(intra_num):
            h = random.randint(min_len, max_len)
            w = random.randint(min_len, max_len)
            mask_region = torch.ones(h, w)

            y = np.arange(H - h + 1)
            x = np.arange(W - w + 1)
            random.shuffle(y)
            random.shuffle(x)
            i = j = None
            for y1 in y:
                for x1 in x:
                    y2 = y1 + h - 1
                    x2 = x1 + w - 1
                    sum1 = integral[idx, y1 - 1, x1 - 1] if x1 * y1 != 0 else 0
                    sum2 = integral[idx, y1 - 1, x2] if y1 != 0 else 0
                    sum3 = integral[idx, y2, x2]
                    sum4 = integral[idx, y2, x1 - 1] if x1 != 0 else 0
                    scores = sum3 + sum1 - sum2 - sum4
                    if scores >= min_num:
                        i, j = int(y1), int(x1)
                        break

                if i is not None:
                    break
            if i is None:   # use the complementary set as intra neg
                # print('h,w', h, w)
                # print(intra_mask_map[idx, k, :, :])
                mask_map_k = np.hstack([
                    np.zeros(C - C//2),
                    np.ones(C//2),
                ])
                np.random.shuffle(mask_map_k)
                mask_map_k = torch.from_numpy(mask_map_k.reshape(H, W))
                intra_mask_map[idx, k, :, :] = intra_mask_map[idx, k, :, :] * mask_map_k.cuda()
            else:
                mask_map_k = torch.zeros(H, W)
                mask_map_k[i:i+h, j:j+w] = mask_region
                intra_mask_map[idx, k, :, :] = intra_mask_map[idx, k, :, :] * mask_map_k.cuda()
                # print('after maxtrix:',intra_mask_map[idx, k, :, :])

    intra_mask_map = intra_mask_map.to(torch.bool).reshape(B, intra_num, -1)
    return intra_mask_map


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'moco_deit_small_patch16_224_V3',
    'moco_beit_small_patch16_224_V3',
    'moco_beit_base_patch16_224_V3',
]


class VisionTransformerForMaskedMOCOV3(nn.Module):
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
        self.shared_proj = cfg.model.shared_proj
        self.proj_num_layers = cfg.model.proj_num_layers
        assert self.proj_num_layers > 0
        self.pred_num_layers = cfg.model.pred_num_layers
        self.use_last_bn = cfg.model.use_last_bn
        self.stop_grad_conv1 = cfg.model.stop_grad_conv1
        self.use_attention = cfg.model.use_attention  # use attention map to generate mask map
        self.use_intra_negative = cfg.model.use_intra_negative
        self.intra_num = 15
        self.decoder_layers = cfg.model.decoder_layers
        self.target_decoder = cfg.model.target_decoder
        if self.target_decoder:
            assert self.decoder_layers > 0
        self.use_cls_feature = cfg.model.use_cls_feature  # only use cls token for pretraining
        self.only_patch_feature = cfg.model.only_patch_feature
        self.use_contrast_lg = cfg.model.use_contrast_lg
        self.use_contrast_gl = cfg.model.use_contrast_gl
        self.mask_ratio = cfg.data.mask_ratio
        self.mask_shape = cfg.data.mask_shape
        self.sincos_emb = cfg.model.sincos_emb
        print('ViT for MaskMOCO  feat_dim:{}, shared_proj:{}, proj_num_layers:{}, pred_num_layers:{}, momentum:{}, temperature:{}, queue_len:{}'\
              .format(self.feat_dim, self.shared_proj, self.proj_num_layers, self.pred_num_layers, self.momentum, self.contrast_temperature, self.queue_len))
        print('use attention map as a guide') if self.use_attention else print('do not use attention map')

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_k = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            if self.sincos_emb:
                print('use sin-cos pos emb\n')
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
                self.pos_embed_k = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                    cls_token=True)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
                self.pos_embed_k.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.pos_embed_k = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
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

        if cfg.model.vanilla is True:
            from models.vision_transformer import Block
        else:
            from models.modeling_finetune import Block
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

        if self.decoder_layers > 0:
            self.decoder = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, window_size=None, attn_head_dim=attn_head_dim, use_groupconv=True,
                )
                for i in range(self.decoder_layers)])
        if self.target_decoder:
            self.decoder_k = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, window_size=None, attn_head_dim=attn_head_dim, use_groupconv=True,
                )
                for i in range(self.decoder_layers)])

        self.norm = norm_layer(embed_dim)
        self.init_std = init_std

        self.projector = MLPHead(in_dim=embed_dim, out_dim=self.feat_dim, num_layers=self.proj_num_layers, last_bn=self.use_last_bn)
        self.projector_k = MLPHead(in_dim=embed_dim, out_dim=self.feat_dim, num_layers=self.proj_num_layers, last_bn=self.use_last_bn)

        if self.shared_proj is not True:
            self.projector_cls = MLPHead(in_dim=embed_dim, out_dim=self.feat_dim, num_layers=self.proj_num_layers, last_bn=self.use_last_bn)
            self.projector_cls_k = MLPHead(in_dim=embed_dim, out_dim=self.feat_dim, num_layers=self.proj_num_layers, last_bn=self.use_last_bn)

        if self.pred_num_layers > 0:
            self.predictor = MLPHead(in_dim=self.feat_dim, out_dim=self.feat_dim, num_layers=self.pred_num_layers, last_bn=self.use_last_bn)

        if self.pos_embed is not None and self.sincos_emb is False:
            trunc_normal_(self.pos_embed, std=self.init_std)
            trunc_normal_(self.pos_embed_k, std=self.init_std)  # the initial pos_embed should be the same?
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.cls_token_k, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)

        # trunc_normal_(self.projector.weight, std=self.init_std)
        # trunc_normal_(self.predictor, std=self.init_std)
        self.apply(self._init_weights)  # apply _init_weights for all submodules, like recursion
        self.fix_init_weight()

        if self.stop_grad_conv1:
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)
            self.patch_embed.proj.weight.requires_grad = False
            self.patch_embed.proj.bias.requires_grad = False

        self._copy_weights(self.patch_embed, self.patch_embed_k)
        self._copy_weights(self.blocks, self.blocks_k)
        self._copy_weights(self.projector, self.projector_k)
        if self.target_decoder:
            self._copy_weights(self.decoder, self.decoder_k)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        if self.pred_num_layers > 0:
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
        if self.shared_proj is not True:
            self._copy_weights(self.projector_cls, self.projector_cls_k)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_cls)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_cls_k)

        self.K = int(1281152 * 1. / dist.get_world_size() / cfg.data.batch_size) * cfg.epochs   # train_data_num
        self.k = int(1281152 * 1. / dist.get_world_size() / cfg.data.batch_size) * cfg.start_epoch

        # create the queue
        self.register_buffer("queue1", torch.randn(self.feat_dim, self.queue_len))
        self.register_buffer("queue2", torch.randn(self.feat_dim, self.queue_len))
        self.register_buffer("queue1_cls", torch.randn(self.feat_dim, self.queue_len))
        self.register_buffer("queue2_cls", torch.randn(self.feat_dim, self.queue_len))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.queue1_cls = F.normalize(self.queue1_cls, dim=0)
        self.queue2_cls = F.normalize(self.queue2_cls, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        if self.decoder_layers > 0:
            for layer_id, layer in enumerate(self.decoder):
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
        return {'pos_embed', 'pos_embed_k', 'cls_token', 'cls_token_k'}

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
        if self.shared_proj is not True:
            for param_q, param_k, in zip(self.projector_cls.parameters(), self.projector_cls_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
        if self.target_decoder:
            for param_q, param_k, in zip(self.decoder.parameters(), self.decoder_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2, keycls1, keycls2):
        # gather keys before updating queue
        keys1 = dist_collect(keys1)
        keys2 = dist_collect(keys2)
        keycls1 = dist_collect(keycls1)
        keycls2 = dist_collect(keycls2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.T
        self.queue2[:, ptr:ptr + batch_size] = keys2.T
        self.queue1_cls[:, ptr:ptr + batch_size] = keycls1.T
        self.queue2_cls[:, ptr:ptr + batch_size] = keycls2.T
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1  # number of patches
        if self.pos_embed is None:
            return 0
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, bool_masked_pos=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # add the [CLS] token to the embed patch tokens
        mask_token = self.mask_token.expand(B, 196, -1)

        # replace the masked visual tokens by mask_token
        if bool_masked_pos is not None:
            m = bool_masked_pos.unsqueeze(-1).type_as(mask_token)  # w.shape ([64, 196, 1]), elements are 0 or 1
            x = x * (1 - m) + mask_token * m

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)  # add positional encoding to each token
        return self.pos_drop(x)

    def get_last_selfattention(self, x, bool_masked_pos=None):
        x = self.prepare_tokens(x, bool_masked_pos=bool_masked_pos)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        print('get last self attention from target network (blocks-k) \n')
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 8:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                return attn

    def contrastive_loss(self, q, k, queue, use_intra=False):
        if use_intra:
            negatives = []
            k_all = k.reshape(-1, self.intra_num + 1, self.feat_dim)
            k = k_all[:, 0, :]
            k_intra_neg = k_all[:, 1:, :]  # [B, K-1, C]
            negatives.append(k_intra_neg.clone().detach())
            queue_negs = queue.clone().detach().transpose(0, 1).unsqueeze(0).repeat(k.shape[0], 1, 1)  # [B, queue_len, dim]
            negatives.append(queue_negs)
            patch_k_neg = torch.cat(negatives, dim=1)
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,nkc->nk', [q, patch_k_neg])
        else:
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
            acc = corrects / N
        return loss, acc

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

            cls_tokens_k = self.cls_token_k.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x_k = torch.cat((cls_tokens_k, x_k), dim=1)
            if self.pos_embed_k is not None:
                x_k = x_k + self.pos_embed_k
            x_k = self.pos_drop(x_k)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

            for i, blk in enumerate(self.blocks_k):
                if i < len(self.blocks_k) - 1:
                    x_k = blk(x_k, rel_pos_bias=rel_pos_bias)
                else:
                    x_k, attn = blk(x_k, rel_pos_bias=rel_pos_bias, return_attention=True)
        return self.norm(x_k), attn

    def cal_mask_tokens_value(self, x, bool_masked_pos):
        '''to calculate the mean value of the masked tokens in an image'''
        x = x[:, 1:]
        int_masked_pos = bool_masked_pos.to(torch.int)  # [True, False...] -> [1,0...], torch.Size([64, 196])
        x = x.transpose(1, 2)  # torch.Size([64, 768, 196])
        int_masked_pos = int_masked_pos.unsqueeze(-1).to(torch.float)  # [64, 196, 1]
        x = (x @ int_masked_pos).squeeze(-1)  # torch.Size([64, 768])
        x = x / (int_masked_pos.sum(1))  # [64, 768] / [64, 1]
        return x

    def forward(self, x_1, x_2, bool_masked_pos_1, bool_masked_pos_2, return_all_tokens=False, mask=True):
        x_k_2, attn_2 = self.forward_features_k(x_2, bool_masked_pos=bool_masked_pos_2)  # mask map not used in fact
        x_k_1, attn_1 = self.forward_features_k(x_1, bool_masked_pos=bool_masked_pos_1)  # mask map not used in fact
        if self.use_attention:
            with torch.no_grad():
                bool_masked_pos_1 = generate_maskmap_from_attn(attentions=attn_1, mask_ratio=self.mask_ratio, mask_shape=self.mask_shape)
                bool_masked_pos_2 = generate_maskmap_from_attn(attentions=attn_2, mask_ratio=self.mask_ratio, mask_shape=self.mask_shape)
                bool_masked_pos_1.to(torch.device('cuda'), non_blocking=True)
                bool_masked_pos_2.to(torch.device('cuda'), non_blocking=True)   # torch.tensor([B, 196])
        x_q_1 = self.forward_features(x_1, bool_masked_pos=bool_masked_pos_1, mask=mask)
        x_q_2 = self.forward_features(x_2, bool_masked_pos=bool_masked_pos_2, mask=mask)

        x_cls_1, x_cls_k_1, x_cls_2, x_cls_k_2 = x_q_1[:, 0], x_k_1[:, 0], x_q_2[:, 0], x_k_2[:, 0]

        if return_all_tokens:
            x_q_1 = x_q_1[:, 1:].mean(1)  # torch.Size([64, 196, 768])  [batch=64, patch_num=14 x 14, patch_embed]
            x_k_1 = x_k_1[:, 1:].mean(1)    # elements in x may less than 0
            x_q_2 = x_q_2[:, 1:].mean(1)
            x_k_2 = x_k_2[:, 1:].mean(1)
        else:
            x_q_1 = self.cal_mask_tokens_value(x_q_1, bool_masked_pos_1)
            x_q_2 = self.cal_mask_tokens_value(x_q_2, bool_masked_pos_2)
            with torch.no_grad():
                x_k_1 = self.cal_mask_tokens_value(x_k_1, bool_masked_pos_1)
                x_k_2 = self.cal_mask_tokens_value(x_k_2, bool_masked_pos_2)

        feat_q_1 = self.projector(x_q_1)      # feat_q torch.Size([64, 256])
        feat_q_2 = self.projector(x_q_2)
        if self.pred_num_layers > 0:
            feat_q_1 = self.predictor(feat_q_1)
            feat_q_2 = self.predictor(feat_q_2)
        feat_q_1 = F.normalize(feat_q_1, dim=1)
        feat_q_2 = F.normalize(feat_q_2, dim=1)

        with torch.no_grad():
            feat_k_1 = F.normalize(self.projector_k(x_k_1), dim=1)
            feat_k_2 = F.normalize(self.projector_k(x_k_2), dim=1)

        if self.shared_proj:
            feat_cls_q_1 = self.projector(x_cls_1)
            feat_cls_q_2 = self.projector(x_cls_2)
            if self.pred_num_layers > 0:
                feat_cls_q_1 = self.predictor(feat_cls_q_1)
                feat_cls_q_2 = self.predictor(feat_cls_q_2)
            feat_cls_q_1 = F.normalize(feat_cls_q_1, dim=1)
            feat_cls_q_2 = F.normalize(feat_cls_q_2, dim=1)
            with torch.no_grad():
                feat_cls_k_1 = F.normalize(self.projector_k(x_cls_k_1), dim=1)
                feat_cls_k_2 = F.normalize(self.projector_k(x_cls_k_2), dim=1)
        else:
            raise NotImplementedError

        if self.use_cls_feature:
            loss_patch1, patch_acc1 = 0, 0
            loss_patch2, patch_acc2 = 0, 0
        else:
            loss_patch1, patch_acc1 = self.contrastive_loss(feat_q_1, feat_k_2, self.queue1, use_intra=self.use_intra_negative)
            loss_patch2, patch_acc2 = self.contrastive_loss(feat_q_2, feat_k_1, self.queue2, use_intra=self.use_intra_negative)
        if self.only_patch_feature:
            loss_cls1, cls_acc1 = 0, 0
            loss_cls2, cls_acc2 = 0, 0
        else:
            loss_cls1, cls_acc1 = self.contrastive_loss(feat_cls_q_1, feat_cls_k_2, self.queue1_cls)
            loss_cls2, cls_acc2 = self.contrastive_loss(feat_cls_q_2, feat_cls_k_1, self.queue2_cls)
        loss = (loss_cls1 + loss_cls2 + loss_patch1 + loss_patch2) * 0.5
        patch_acc = (patch_acc1 + patch_acc2) * 0.5
        cls_acc = (cls_acc1 + cls_acc2) * 0.5

        self._dequeue_and_enqueue(feat_k_2, feat_k_1, feat_cls_k_2, feat_cls_k_1)
        return loss, (cls_acc, patch_acc)


@register_model
def moco_deit_small_patch16_224_V3(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedMOCOV3(
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
def moco_beit_small_patch16_224_V3(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedMOCOV3(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def moco_beit_base_patch16_224_V3(pretrained=False, **kwargs):
    _ = kwargs.pop("num_classes")
    model = VisionTransformerForMaskedMOCOV3(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


