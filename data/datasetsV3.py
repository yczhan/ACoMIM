'''
dataset V3 for masked MOCO, mask with resize,
'''

import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
import math
import warnings
from collections.abc import Sequence

from torchvision import datasets, transforms
import torchvision.transforms.functional as F

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from data.transforms import RandomResizedCropAndInterpolationWithTwoPic, RandomResizedCropAndFlipBothTwoImg
from timm.data import create_transform

from dall_e.utils import map_pixels
from data.masking_generator import MaskingGenerator, RandomMaskingGenerator
from data.dataset_folder import ImageFolder, ClassificationDataset, ListDataset


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def box2maskmap(img_size, box_param):
    '''
    Args:
        img_size: the image size, (h, w)
        box_param: the parameters of the mask box, (i, j, h, w)
    Returns:
        mask map: np array int, [img_size[0], img_size[1]]
    '''
    if isinstance(img_size, int):  # assert that the original image is a square, e.g. 25 x 25
        img_size = (img_size, img_size)
    elif not isinstance(img_size, tuple):
        raise NotImplementedError
    img_h, img_w = img_size
    box_i, box_j, box_h, box_w = box_param

    mask_map = np.zeros(shape=(img_w, img_h), dtype=int)
    for w in range(box_w):
        for h in range(box_h):
            mask_map[box_i + h, box_j + w] = 1
    return mask_map


def maskmap2box(mask_map):
    '''
    Args:
        mask map: np array int, [img_size[0], img_size[1]]
    Returns:
        mask box: (i, j, h, w)
    '''
    column = (mask_map != 0).argmax(axis=0)
    row = (mask_map != 0).argmax(axis=1)
    if np.nonzero(column)[0].size == 0:
        i = 0
        w = max((mask_map != 0).sum(axis=1))    # in case 
    else:
        i = min(column[np.nonzero(column)])  # in case column is all zero
        w = np.count_nonzero(column)
    if np.nonzero(row)[0].size == 0:
        j = 0
        h = max((mask_map != 0).sum(axis=0))
    else:
        j = min(row[np.nonzero(row)])
        h = np.count_nonzero(row)

    return i, j, h, w


def select_mask_region(side_len=25, min_num=39, max_num=78, importance=None):
    """
    :param side_len: the length of image side (how many tokens)
    :param min_num: the minimum number of mask tokens, 20% by default
    :param max_num: the maximun number of mask tokens, 40% by default
    :param importance: the importance map of the image, grad_cam by default
    :return: the parameters of mask box (i, j, h, w) ; i, j is the coordinates of corner
    """
    if isinstance(side_len, int):  # assert that the original image is a square, e.g. 25 x 25
        side_len = (side_len, side_len)
    elif not isinstance(side_len, tuple):
        raise NotImplementedError
    min_len = math.ceil(math.ceil(min_num / 2) ** 0.5)  # assert that the box ratio is 0.5 ~ 2
    max_len = int((max_num * 2) ** 0.5)  # min & max length of box side
    if importance is None:
        for attempt in range(10):
            h = random.randint(min_len, max_len)
            w = random.randint(min_len, max_len)
            if 0.5 * w < h < 2 * w and min_num < h * w < max_num and w < side_len[0] and h < side_len[1]:
                j = random.randint(0, side_len[0] - w)  # random.randint(a, b), both a,b are inclusive
                i = random.randint(0, side_len[1] - h)
                return i, j, h, w
            else:
                continue
        # Fallback to central mask
        h = w = math.ceil(((min_num + max_num) / 2) ** 0.5)
        j = int((side_len[0] - w) / 2)
        i = int((side_len[0] - h) / 2)
        return i, j, h, w
    else:
        importance = F.resize(img=importance, size=side_len)  # resize the importance map to token format (e.g. 25x25)
        # calculate integral map to accelerate sum operation
        importance_np = np.array(importance)
        min_value = importance_np.min()
        max_value = importance_np.max()
        importance_np = (importance_np - min_value) / max(max_value - min_value, 0.001)
        integral = np.cumsum(np.cumsum(importance_np, axis=1), axis=0)

        for attempt in range(10):
            h = random.randint(min_len, max_len)
            w = random.randint(min_len, max_len)
            if 2 * w > h > 0.5 * w and max_num > h * w > min_num and w < side_len[0] and h < side_len[1]:
                break
            elif attempt == 9:
                h = w = math.ceil(((min_num + max_num) / 2) ** 0.5)

        # propose dozens of bounding boxes, 20 boxes by default
        y1 = torch.randint(0, side_len[1] - h + 1, size=(20,)).numpy()  # side_len[1] - h + 1 is exclusive
        x1 = torch.randint(0, side_len[0] - w + 1, size=(20,)).numpy()
        y2 = y1 + h - 1
        x2 = x1 + w - 1

        sum1 = integral[y1, x1]
        sum2 = integral[y1, x2]
        sum3 = integral[y2, x2]
        sum4 = integral[y2, x1]
        scores = sum3 + sum1 - sum2 - sum4

        sorted_inds = np.argsort(scores)[::-1]
        keep_lens = max(int(len(sorted_inds) * 0.1), 1)
        select_idx = int(sorted_inds[random.randint(0, keep_lens - 1)])
        i, j = int(y1[select_idx]), int(x1[select_idx])
        return i, j, h, w


def proj_sr2img(sr_mask_map, sr_i, sr_j, sr_h, sr_w, out_size):
    '''
    Args:
        sr_mask_map: np array int, the mask map in the search region, we need to project it to the original image
        sr_i, j, h ,w: the crop parameters of search region in the original image
        out_size: output height & width, that is the original image's size
    Returns: np array int (out_size), the mask map in the original image
    '''
    if isinstance(out_size, (int, float)):
        out_size = (out_size, out_size)
    # copy to avoid inplace op
    sr_mask_map = sr_mask_map.copy()
    sr_mask_map = torch.from_numpy(sr_mask_map).float().unsqueeze(0)
    sr_mask_map = F.resize(sr_mask_map, size=(sr_h, sr_w))
    sr_mask_map = (sr_mask_map + 1e-4).round().int()  # 1e-4 is to make sure 0.5 -> 1 instead of 0
    mask_map = np.zeros(shape=out_size, dtype=int)
    mask_map[sr_i:sr_i + sr_h, sr_j:sr_j + sr_w] = sr_mask_map
    return mask_map


class DataAugmentationForMaskMOCOV3(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.patch_size = args.patch_size[0]
        self.pre_size = 400
        self.size = (args.input_size, args.input_size)  # (224, 224) by default
        self.max_mask_num = int(((args.input_size // self.patch_size) ** 2) * min(args.mask_ratio + 0.1,
                                                                                  1.0))  # mask ratio range: (0.3, 0.5) by default
        self.min_mask_num = int(((args.input_size // self.patch_size) ** 2) * max(args.mask_ratio - 0.1, 0))

        if args.use_grad_cam:
            self.common_transform = RandomResizedCropAndFlipBothTwoImg(size=self.pre_size,
                                                                       ratio=(4. / 5., 5. / 4.),
                                                                       interpolation=args.train_interpolation,
                                                                       scale=(0.5, 1.0))

        else:
            self.common_transform = RandomResizedCropAndInterpolationWithTwoPic(size=self.pre_size,
                                                                                ratio=(4. / 5., 5. / 4.),
                                                                                interpolation=args.train_interpolation,
                                                                                scale=(0.5, 1.0))
        # after common transform, the image is 400 x 400, that is 25 x 25 patches

        self.discrete_resizedcrop = DiscreteResizeCropAndFlipWithMaskRegion(size=args.input_size,
                                                                            patch_size=args.patch_size)

        self.transform_1 = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        self.transform_2 = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
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

    def __call__(self, image, importance=None):
        if importance is None:
            pre_img = self.common_transform(image)
        else:
            pre_img, importance = self.common_transform(image, importance)

        i, j, h, w, flip = self.discrete_resizedcrop.get_params(img=pre_img)  # i,j,h,w are patch-wise
        img_q = F.resized_crop(img=pre_img, top=i * self.patch_size, left=j * self.patch_size,
                               height=h * self.patch_size, width=w * self.patch_size, size=self.size)
        if importance is not None:
            importance = F.resized_crop(img=importance, top=i * self.patch_size, left=j * self.patch_size,
                                        height=h * self.patch_size, width=w * self.patch_size, size=self.size)
        # select mask_box in query region
        mask_box_q = select_mask_region(side_len=self.size[0] // self.patch_size, min_num=self.min_mask_num,
                                        max_num=self.max_mask_num, importance=importance)

        mask_map_q = box2maskmap(img_size=self.size[0] // self.patch_size, box_param=mask_box_q)  # mask map in search image
        mask_region = proj_sr2img(mask_map_q, sr_i=i, sr_j=j, sr_h=h, sr_w=w,
                                  out_size=self.pre_size // self.patch_size)  # mask map in original image
        mask_box = maskmap2box(mask_region)  # mask box in original image
        img_k, mask_map_k = self.discrete_resizedcrop(img=pre_img, mask_box=mask_box)

        if flip:
            img_q = F.hflip(img_q)
            mask_map_q = np.fliplr(mask_map_q)
            mask_map_q = np.ascontiguousarray(mask_map_q)

        sample_q = self.transform_1(img_q)
        sample_k = self.transform_2(img_k)

        return sample_q, sample_k, mask_map_q, mask_map_k

    def __repr__(self):
        repr = "(DataAugmentationForMaskMOCOV3, random discrete resized crop\n"
        repr += "  number of mask tokens : (%s, %s),\n" % (self.min_mask_num, self.max_mask_num)
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  discrete_resizedcrop = %s,\n" % str(self.discrete_resizedcrop)
        repr += "  transform1 for q = %s,\n" % str(self.transform_1)
        repr += "  transform2 for k = %s,\n" % str(self.transform_2)
        repr += ")"
        return repr


class DiscreteResizeCropAndFlipWithMaskRegion:
    """Discrete Crop and resize the given PIL Image
    if mask_box exists, then crop & resize it in the same way.
    A resized crop of discrete size (default: 224, e.g. 14 x 14 tokens)
    random flip the img (and mask map)
    Args:
        size: expected output size of each edge after crop
        patch_size: (16, 16) by default, the side length of a patch
    return:
        image (and the corresponding int mask map)
    """

    def __init__(self, size=224, patch_size=16, scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if isinstance(patch_size, tuple):
            self.patch_size = patch_size[0]
        else:
            self.patch_size = int(patch_size)
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, mask_box=None):
        """Get parameters for ``crop`` for discrete resized crops.
        if mask_box exists, then crop & resize it in the same way.
        Note that the mask region after the crop should contain the most of the original

        Args:
            img (PIL Image): Image to be cropped.
            mask_box: the mask region in pre_img (i,j,h,w)

        Returns:
            i, j, h, w to be passed to ``crop`` for a discrete crop (patch-wise)
                Note: x=i, y=j is the coordinates of top left corner
            flip: True or False
        """
        img_w = img.size[0] // self.patch_size  # 25 by default
        img_h = img.size[1] // self.patch_size
        area = img_w * img_h
        if mask_box is not None:
            box_i, box_j, box_h, box_w = mask_box  # [0 ~ img_w - 1]

        for tmp in range(11):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            log_ratio = torch.log(torch.tensor(self.ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if tmp == 10:  # Fallback to central crop
                in_ratio = float(img_w) / float(img_h)
                if in_ratio < min(self.ratio):
                    crop_w = img_w
                    crop_h = int(round(crop_w / min(self.ratio)))
                elif in_ratio > max(self.ratio):
                    crop_h = img_h
                    crop_w = int(round(crop_h * max(self.ratio)))
                else:  # whole image
                    crop_w = img_w
                    crop_h = img_h

            if mask_box is None:  # normal crop & resize
                if 0 < crop_w <= img_w and 0 < crop_h <= img_h:
                    crop_i = torch.randint(0, img_h - crop_h + 1, size=(1,)).item()
                    crop_j = torch.randint(0, img_w - crop_w + 1, size=(1,)).item()
                    break
            else:  # we have to cover the mask region while cropping
                crop_w = max(crop_w, box_w)  # make sure the crop region can cover the mask box
                crop_h = max(crop_h, box_h)
                if 0 < crop_w <= img_w and 0 < crop_h <= img_h:
                    crop_i = random.randint(max(0, box_i + box_h - crop_h), min(box_i, img_h - crop_h))
                    crop_j = random.randint(max(0, box_j + box_w - crop_w), min(box_j, img_w - crop_w))
                    break

        flip = random.random()
        flip = True if flip > 0.5 else False

        return crop_i, crop_j, crop_h, crop_w, flip

    def __call__(self, img, mask_box=None):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
            mask_box: the mask region: (i, j, h, w)  token-wise
        Returns:
            PIL Image: Randomly cropped and resized image. (and corresponding int mask map)
        """
        i, j, h, w, flip = self.get_params(img, mask_box)  # i,j,h,w are patch-wise
        sample = F.resized_crop(img=img, top=i * self.patch_size, left=j * self.patch_size, height=h * self.patch_size,
                             width=w * self.patch_size, size=self.size)
        if flip:
            sample = F.hflip(sample)

        if mask_box is not None:
            img_w = img.size[0] // self.patch_size  # 25 by default
            img_h = img.size[1] // self.patch_size
            pre_mask_map = box2maskmap(img_size=(img_h, img_w), box_param=mask_box)
            pre_mask_map = torch.from_numpy(pre_mask_map).float().unsqueeze(0)
            mask_map = F.resized_crop(img=pre_mask_map, top=i, left=j, height=h, width=w,
                                      size=(self.size[0] // self.patch_size, self.size[1] // self.patch_size))
            mask_map = mask_map.round().int()  # convert to int mask map (0 and 1)
            if flip:
                mask_map = F.hflip(mask_map)
            mask_map = mask_map.numpy().squeeze(0)  # convert to np array (h, w) from tensor (1, h, w)
            return sample, mask_map
        else:
            return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}, patch_size={1}'.format(self.size, self.patch_size)
        format_string += ')'
        return format_string


def build_moco_pretraining_dataset(args):
    transform = DataAugmentationForMaskMOCOV3(args)
    if args.use_grad_cam is not True:
        print("Data Aug = %s" % str(transform))
        if args.data_set == 'IMNET':
            root = os.path.join(args.data_path, 'train')
            txt_path = os.path.join(args.data_path, 'train_map.txt')
            return ClassificationDataset(root=root, transform=transform, txt_path=txt_path)
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
    else:
        raise NotImplementedError
