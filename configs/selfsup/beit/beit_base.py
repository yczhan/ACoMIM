
# base settings for beit
epochs = 100
save_ckpt_freq = 1
update_freq = 1
start_epoch = 0
auto_resume = True
resume = None
model_ema = False

seed = 0
device = 'cuda'
pin_mem = True  # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
data_set = 'IMNET'
data_path = '/mnt/data/'
discrete_vae_weight_path = '/mnt/yuczhan/dall_e'
discrete_vae_type = "dall-e"

num_mask_patches = 75
max_mask_patches_per_block = None
min_mask_patches_per_block = 16

# model settings
model = dict(
    type='BEIT',
    model='beit_base_patch16_224_8k_vocab',
    pretrained=None,
    num_mask_patches=num_mask_patches,
    max_mask_patches_per_block=max_mask_patches_per_block,
    min_mask_patches_per_block=min_mask_patches_per_block,
    mask=True,
    return_all_tokens=False,
    drop_path=0.1,
    rel_pos_bias=False,
    shared_rel_pos_bias=False,
    abs_pos_emb=True,
    layer_scale_init_value=None,
    model_ema=model_ema,
)

# optimizer
optimizer = dict(opt='adamw', opt_eps=1e-8, lr=1.5e-3, warmup_lr=1e-6, min_lr=1e-5, weight_decay=0.05, momentum=0.9,
                 warmup_epochs=10, warmup_steps=-1, weight_decay_end=None, clip_grad=3.0,)

data = dict(
    data_set=data_set,
    imagenet_default_mean_and_std=True,
    batch_size=64,  # total 64*8 (*update_freq)
    num_workers=10,
    data_path=data_path,
    input_size=224,
    second_input_size=112,
    discrete_vae_weight_path=discrete_vae_weight_path,
    discrete_vae_type=discrete_vae_type,
    drop_last=True,
    num_mask_patches=num_mask_patches,
    max_mask_patches_per_block=max_mask_patches_per_block,
    min_mask_patches_per_block=min_mask_patches_per_block,

    color_jitter=0.4,
    random_resized_crop=dict(size=224, scale=(0.2, 1.)),
    train_interpolation='bicubic',
    second_interpolation='lanczos',
    # for mlmcls
    reprob=0.25,
    remode='pixel',
    recount=1,
    resplit=False,
    crop_pct=None,
    aa='rand-m9-mstd0.5-inc1',
    nb_classes=1000,
    # boxes_per_img=16,
    # box_scale_range=(42, 128),
    # box_ratio_range=(-0.69, 0.69),
    # pos_iou_threshold=0.8,
    # neg_iou_threshold=0.2,
    # patch_query_type='roi',
)




