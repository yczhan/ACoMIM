# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import datetime
import numpy as np
import random
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from mmcv import Config

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer

from engine_for_pretraining import train_one_epoch_contrast
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models.modeling_pretrain
import models.modeling_pretrainV3



def get_args():
    parser = argparse.ArgumentParser('Mask ViT contrast pre-training script', add_help=False)
    parser.add_argument('config', help='train config file path')  ###
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        model_name=args.model.model,
        pretrained=False,
        drop_path_rate=args.model.drop_path,
        drop_block_rate=None,
        use_rel_pos_bias=args.model.rel_pos_bias,
        use_shared_rel_pos_bias=args.model.shared_rel_pos_bias,
        use_abs_pos_emb=args.model.abs_pos_emb,
        use_groupconv=args.model.use_groupconv,
        init_values=args.model.layer_scale_init_value,
        cfg=args,
    )

    return model


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True


    from data.datasetsV3 import build_moco_pretraining_dataset

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.data.window_size = (args.data.input_size // patch_size[0], args.data.input_size // patch_size[1])
    args.data.patch_size = patch_size

    # get dataset
    if args.data_set == 'IMNET' or args.data_set == 'TINY-IMNET':
        args.data.data_path = os.path.join(args.data_path, 'ImageNet')
    elif args.data_set == 'COCO':
        args.data.data_path = os.path.join(args.data_path, 'coco')
    elif args.data_set == 'CC':
        args.data.data_path = os.path.join(args.data_path, 'conceptual-captions')
    else:
        raise NotImplementedError

    if args.data.use_grad_cam:
        args.data.grad_cam_path = os.path.join(args.data.data_path, 'grad_cam_layer4_2')
        if os.path.exists(args.data.grad_cam_path):
            print('\n Use grad cam maps as a guide : {} \n'.format(args.data.grad_cam_path))
        else:
            raise Exception('No grad cam files found in {}'.format(args.data.grad_cam_path))

    if args.data.moby_aug:
        raise NotImplementedError
    else:
        dataset_train = build_moco_pretraining_dataset(args.data)
    print('#########################################\n pretraining args: \n', args,
          '\n#########################################\n')

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.data.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.data.batch_size,
        num_workers=args.data.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # for batch in enumerate(data_loader_train):
    #     print(batch)    # (0, samples, images, mask_bool...
    #     exit(0)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.data.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.optimizer.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args.optimizer, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.optimizer.lr, args.optimizer.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.optimizer.warmup_epochs, warmup_steps=args.optimizer.warmup_steps,
    )
    if args.optimizer.weight_decay_end is None:
        args.optimizer.weight_decay_end = args.optimizer.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.optimizer.weight_decay, args.optimizer.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch_contrast(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.optimizer.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq, mask=args.model.mask, return_all_tokens=args.model.return_all_tokens
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            else:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, epoch_name='latest')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()
    cfg = Config.fromfile(args.config)
    cfg.world_size = args.world_size
    cfg.local_rank = args.local_rank
    cfg.dist_url = args.dist_url
    cfg.dist_on_itp = args.dist_on_itp

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        cfg.output_dir = args.output_dir
    else:
        cfg_file_path = args.config.split('configs/')[-1].split('.')[0]  # '/selfsup/mcc/mcc_debug'
        cfg.output_dir = os.path.join(cfg.output_dir, cfg_file_path)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    cfg.log_dir = args.log_dir if args.log_dir is not None else cfg.output_dir

    main(cfg)
