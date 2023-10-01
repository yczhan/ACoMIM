# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from data.transforms import RandomResizedCropAndInterpolationWithTwoPic, _pil_interp
from timm.data import create_transform
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from timm.data.random_erasing import RandomErasing

from dall_e.utils import map_pixels
from data.masking_generator import MaskingGenerator
from data.dataset_folder import ImageFolder, ClassificationDataset, ListDataset


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForMOBY(object):
    def __init__(self, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        sample = self.transform_1(image)
        sample_k = self.transform_2(image)
        return sample, sample_k, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForMOBY,\n"
        repr += "  transform1 = %s,\n" % str(self.transform_1)
        repr += "  transform2 = %s,\n" % str(self.transform_2)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForMaskMOCO(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(size=args.input_size, interpolation=args.train_interpolation),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches = self.common_transform(image)
        sample = self.patch_transform(for_patches)
        sample_k = sample.clone()
        return sample, sample_k, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForMaskMOCO,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForMLMCLS(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.input_size = args.input_size
        self.second_input_size = args.second_input_size
        self.color_jitter = args.color_jitter
        self.auto_augment = args.aa
        self.interpolation = args.train_interpolation
        self.second_interpolation = args.second_interpolation
        self.re_prob = args.reprob
        self.re_mode = args.remode
        self.re_count = args.recount
        self.re_num_splits = 0
        self.mean = mean
        self.std = std
        self.scale = tuple((0.08, 1.0))  # default imagenet scale range
        self.ratio = tuple((3. / 4., 4. / 3.))  # default imagenet ratio range
        self.hflip = 0.5
        self.vflip = 0.

        if isinstance(self.second_input_size, tuple):
            self.second_input_size = self.second_input_size
        else:
            self.second_input_size = (self.second_input_size, self.second_input_size)
        self.second_interpolation = _pil_interp(self.second_interpolation)

        primary_tfl = [RandomResizedCropAndInterpolation(self.input_size, scale=self.scale, ratio=self.ratio, interpolation=self.interpolation)]
        if self.hflip > 0.:
            primary_tfl += [transforms.RandomHorizontalFlip(p=self.hflip)]
        if self.vflip > 0.:
            primary_tfl += [transforms.RandomVerticalFlip(p=self.vflip)]

        secondary_tfl = []
        assert isinstance(self.auto_augment, str)
        if isinstance(self.input_size, tuple):
            img_size_min = min(self.input_size)
        else:
            img_size_min = self.input_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if self.interpolation and self.interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(self.interpolation)
        if self.auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(self.auto_augment, aa_params)]
        elif self.auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(self.auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(self.auto_augment, aa_params)]

        final_tfl = []
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        # if self.re_prob > 0.:
        #     final_tfl.append(
        #         RandomErasing(self.re_prob, mode=self.re_mode, max_count=self.re_count, num_splits=self.re_num_splits, device='cpu'))

        self.common_transform = transforms.Compose(primary_tfl + secondary_tfl)
        self.patch_transform = transforms.Compose(final_tfl)

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches = self.common_transform(image)
        for_visual_tokens = F.resize(img=for_patches, size=self.second_input_size, interpolation=self.second_interpolation)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForMLMCLS,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_beit_pretraining_dataset(args):
    transform = DataAugmentationForBEiT(args)
    print("Data Aug = %s" % str(transform))
    if args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train')
        return ImageFolder(root, transform=transform)
    elif args.data_set == 'TINY-IMNET':
        root = os.path.join(args.data_path, 'train')
        txt_path = os.path.join(args.data_path, 'tiny_train_map.txt')
        return ClassificationDataset(root=root, transform=transform, txt_path=txt_path)
    elif args.data_set == 'COCO':
        root = os.path.join(args.data_path, 'train2017')
        txt_path = os.path.join(args.data_path, 'coco_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)
    elif args.data_set == 'CC':
        root = os.path.join(args.data_path, 'images')
        txt_path = os.path.join(args.data_path, 'caption_train_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)


def build_moco_pretraining_dataset(args):
    transform = DataAugmentationForMaskMOCO(args)
    print("Data Aug = %s" % str(transform))
    if args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train')
        return ImageFolder(root, transform=transform)
    elif args.data_set == 'TINY-IMNET':
        root = os.path.join(args.data_path, 'train')
        txt_path = os.path.join(args.data_path, 'tiny_train_map.txt')
        return ClassificationDataset(root=root, transform=transform, txt_path=txt_path)
    elif args.data_set == 'COCO':
        root = os.path.join(args.data_path, 'train2017')
        txt_path = os.path.join(args.data_path, 'coco_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)
    elif args.data_set == 'CC':
        root = os.path.join(args.data_path, 'images')
        txt_path = os.path.join(args.data_path, 'caption_train_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)


def build_moby_pretraining_dataset(args):
    transform = DataAugmentationForMOBY(args)
    print("Data Aug = %s" % str(transform))
    if args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train')
        return ImageFolder(root, transform=transform)
    elif args.data_set == 'TINY-IMNET':
        root = os.path.join(args.data_path, 'train')
        txt_path = os.path.join(args.data_path, 'tiny_train_map.txt')
        return ClassificationDataset(root=root, transform=transform, txt_path=txt_path)
    elif args.data_set == 'COCO':
        root = os.path.join(args.data_path, 'train2017')
        txt_path = os.path.join(args.data_path, 'coco_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)
    elif args.data_set == 'CC':
        root = os.path.join(args.data_path, 'images')
        txt_path = os.path.join(args.data_path, 'caption_train_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)


def build_mlmcls_pretraining_dataset(args):
    transform = DataAugmentationForMLMCLS(args)
    print("Data Aug = %s" % str(transform))
    if args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train')
        return ImageFolder(root, transform=transform)
    elif args.data_set == 'TINY-IMNET':
        root = os.path.join(args.data_path, 'train')
        txt_path = os.path.join(args.data_path, 'tiny_train_map.txt')
        return ClassificationDataset(root=root, transform=transform, txt_path=txt_path)
    elif args.data_set == 'COCO':
        root = os.path.join(args.data_path, 'train2017')
        txt_path = os.path.join(args.data_path, 'coco_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)
    elif args.data_set == 'CC':
        root = os.path.join(args.data_path, 'images')
        txt_path = os.path.join(args.data_path, 'caption_train_list.txt')
        return ListDataset(root=root, txt_path=txt_path, transform=transform)


def build_dataset(is_train, args):
    if args.linear_aug:
        transform = build_transform_linear_eval(is_train, args)
    else:
        transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if is_train:
            root = os.path.join(args.data_path, 'train')
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            root = os.path.join(args.data_path, 'val')
            txt_path = os.path.join(args.data_path, 'val_map.txt')
            dataset = ClassificationDataset(root=root, txt_path=txt_path, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'TINY-IMNET':
        if is_train:
            root = os.path.join(args.data_path, 'train')
            txt_path = os.path.join(args.data_path, 'tiny_train_map.txt')
            dataset = ClassificationDataset(root=root, txt_path=txt_path, transform=transform)
        else:
            root = os.path.join(args.data_path, 'val')
            txt_path = os.path.join(args.data_path, 'tiny_val_map.txt')
            dataset = ClassificationDataset(root=root, txt_path=txt_path, transform=transform)
        nb_classes = 200
    elif args.data_set == 'COCO':
        raise NotImplementedError
    elif args.data_set == 'CC':
        raise NotImplementedError
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d \n" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        print('\nBuild transform for training:')
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    print('\nBuild transform for val/test:')
    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform_linear_eval(is_train, args):
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        print('\nBuild linear eval transform for training:')
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        print('\nBuild linear eval transform for val/test:')
        transform = transforms.Compose([
            transforms.Resize(args.input_size + 32),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transform


