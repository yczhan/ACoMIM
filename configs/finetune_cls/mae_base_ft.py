# base settings for beit finetune for classification
epochs = 100
save_ckpt = True
save_ckpt_freq = 100
update_freq = 1
start_epoch = 0

finetune = None
load_teacher = False

eval = False  # Perform evaluation only
dist_eval = False  # Enabling distributed evaluation
disable_eval_during_finetuning = False
auto_resume = False
resume = None
model_ema = False

# * Finetuning params
model_key = 'model|module'
model_prefix = ''
init_scale = 0.001
use_mean_pooling = True
# ('--use_cls', action='store_false', dest='use_mean_pooling')
disable_weight_decay_on_rel_pos_bias = False

seed = 0
device = 'cuda'
pin_mem = True  # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
data_set = 'IMNET'
data_path = '/mnt/data/'
nb_classes = 1000
eval_data_path = None

crop_pct = None

# * Random Erase params
reprob = 0.25  # help='Random erase prob (default: 0.25)')
remode = 'pixel'  # help='Random erase mode (default: "pixel")')
recount = 1  # help='Random erase count (default: 1)')
resplit = False  # help='Do not random erase first (clean) augmentation split')

# * Mixup params
mixup = 0.8  # help='mixup alpha, mixup enabled if > 0.')
cutmix = 1.0  # help='cutmix alpha, cutmix enabled if > 0.')
cutmix_minmax = None
# parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
mixup_prob = 1.0  # help='Probability of performing mixup or cutmix when either/both is enabled')
mixup_switch_prob = 0.5  # help='Probability of switching to cutmix when both mixup and cutmix enabled')
mixup_mode = 'batch'  # help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# model settings
model = dict(
    type='BEIT',
    vanilla=False,
    model='beit_base_patch16_224',
    pretrained=None,
    nb_classes=nb_classes,
    drop_path=0.1,
    drop=0.0,
    attn_drop_rate=0.0,
    use_mean_pooling=use_mean_pooling,
    init_scale=init_scale,
    rel_pos_bias=True,
    abs_pos_emb=False,
    use_groupconv=False,
    layer_scale_init_value=0.1,
    disable_eval_during_finetuning=False,
    model_ema=model_ema,
    model_ema_decay=0.9999,
    model_ema_force_cpu=False,

)

# optimizer
optimizer = dict(opt='adamw', opt_eps=1e-8, lr=2e-3, warmup_lr=1e-6, min_lr=1e-6, weight_decay=0.05, momentum=0.9,
                 warmup_epochs=5, warmup_steps=-1, weight_decay_end=None, clip_grad=None, layer_decay=0.75)
# Note that lr = 1e-3 for batch size = 256

data = dict(
    data_set=data_set,
    nb_classes=nb_classes,
    imagenet_default_mean_and_std=True,
    batch_size=64,  # total 64*8 (*update_freq)
    update_freq=update_freq,
    num_workers=10,
    data_path=data_path,
    input_size=224,
    drop_last=True,

    color_jitter=0.4,
    random_resized_crop=dict(size=224, scale=(0.2, 1.)),
    reprob=reprob,
    remode=remode,
    recount=recount,
    resplit=resplit,
    crop_pct=crop_pct,
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
