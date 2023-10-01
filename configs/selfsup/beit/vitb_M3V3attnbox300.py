_base_ = './moco_base.py'
# base settings for beit
epochs = 300
data_set = 'IMNET'
resume = '//ptd6743cbb42acce695b59abe86e3de577/projects/ssl_yczhan/0408exp_pt300/pt-results/ssl_yczhan-bd07588e-0408exp_pt300-s_par_0408vvitb_M3V3attnbox300-d972e8a4/selfsup/beit/0408vvitb_M3V3attnbox300/checkpoint-latest.pth'

# model settings
model = dict(
    type='MOCO',
    vanilla=True,
    layer_scale_init_value=None,
    model='moco_beit_base_patch16_224_V3',
    rel_pos_bias=False,
    abs_pos_emb=True,
    sincos_emb=True,
    shared_rel_pos_bias=False,
    use_groupconv=False,    # when using group conv, other position coding will/should be False
    proj_num_layers=2,  # default = 2,
    pred_num_layers=2,
    shared_proj=True,  # share the projector head in cls/patch token
    use_attention=True,  # use attention map to guide the mask
    momentum=0.999,
    cos_moment=False,
)

# optimizer
optimizer = dict(opt='adamw', opt_eps=1e-8, lr=3e-4, warmup_lr=1e-6, min_lr=1e-5, weight_decay=0.05, momentum=0.9,
                 warmup_epochs=40, warmup_steps=-1, weight_decay_end=None, clip_grad=3.0,)

data = dict(
    type='V3',
    mask_shape='box',
    use_grad_cam=False, # use grad cam to guide crop/mask ,etc
    mask_ratio=0.3,
    data_set=data_set,
    batch_size=64,
)