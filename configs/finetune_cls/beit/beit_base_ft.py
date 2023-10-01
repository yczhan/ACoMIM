_base_ = '../mae_base_ft.py'

# base settings for beit finetune for classification
epochs = 100
save_ckpt = True
save_ckpt_freq = 100
update_freq = 1
start_epoch = 0
data_path = '/mnt/data/'

use_mean_pooling = False


# model settings
model = dict(
    type='BEIT',
    vanilla=True,
    model='beit_base_patch16_224',
    drop_path=0.2,
    use_mean_pooling=use_mean_pooling,
    pretrained=None,
    rel_pos_bias=False,
    abs_pos_emb=True,
    use_groupconv=False,
    layer_scale_init_value=None,
)

# optimizer
optimizer = dict(opt='adamw', opt_eps=1e-8, lr=4e-3, warmup_lr=1e-6, min_lr=1e-6, weight_decay=0.05, momentum=0.9,
                 warmup_epochs=20, warmup_steps=-1, weight_decay_end=None, clip_grad=None, layer_decay=0.65)
# Note that lr = 1e-3 for batch size = 256


data = dict(
    imagenet_default_mean_and_std=True,
    batch_size=128,  # total 64*8 (*update_freq)
    update_freq=update_freq,
    num_workers=10,
    data_path=data_path,
    input_size=224,
    drop_last=True,

    color_jitter=0.4,
    random_resized_crop=dict(size=224, scale=(0.2, 1.)),

    aa='rand-m9-mstd0.5-inc1',
    # help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'
    smoothing=0.1,
    train_interpolation='bicubic',
    linear_aug=False,
    boxes_per_img=16,
    box_scale_range=(42, 128),
    box_ratio_range=(-0.69, 0.69),
    pos_iou_threshold=0.8,
    neg_iou_threshold=0.2,
    patch_query_type='roi',
)
