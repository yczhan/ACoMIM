_base_ = '../mae_base_ft.py'

# base settings for beit finetune for classification
epochs = 100
save_ckpt = True
save_ckpt_freq = 100
update_freq = 1
start_epoch = 0

seed = 42


# model settings
model = dict(
    type='BEIT',
    vanilla=True,
    model='beit_base_patch16_224',
    pretrained=None,
    rel_pos_bias=False,
    abs_pos_emb=True,
    use_groupconv=False,
    layer_scale_init_value=None,
)

# optimizer
optimizer = dict(opt='adamw', opt_eps=1e-8, lr=2e-3, warmup_lr=1e-6, min_lr=1e-6, weight_decay=0.05, momentum=0.9,
                 warmup_epochs=5, warmup_steps=-1, weight_decay_end=None, clip_grad=None, layer_decay=0.75)
# Note that lr = 1e-3 for batch size = 256


