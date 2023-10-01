# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------'
import torch
import torchvision.transforms.functional as F
from PIL import Image
import warnings
import math
import random
import numpy as np


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


class RandomResizedCropAndInterpolationWithTwoPic:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear', second_interpolation='lanczos'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = _pil_interp(second_interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation), \
                   F.resized_crop(img, i, j, h, w, self.second_size, self.second_interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0}'.format(interpolate_str)
        if self.second_size is not None:
            format_string += ', second_size={0}'.format(self.second_size)
            format_string += ', second_interpolation={0}'.format(_pil_interpolation_to_str[self.second_interpolation])
        format_string += ')'
        return format_string


class DiscreteCropAndFlipWithMaskRegion:
    """Discrete Crop the given PIL Image according to the mask region. return 2 views and 2 mask map
    A crop of discrete size (default: 224, e.g. 14 x 14 tokens)
    random flip the img and mask map
    Args:
        size: expected output size of each edge after crop
        patch_size: (16, 16) by default, the side length of a patch
    return:
        two images and the corresponding mask map
    """

    def __init__(self, size=224, patch_size=16):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if isinstance(patch_size, tuple):
            self.patch_size = patch_size[0]
        else:
            self.patch_size = int(patch_size)

    def get_params(self, img, mask_box):
        """Get parameters for ``crop`` for two discrete crops and the corresponding mask maps.

        Args:
            img (PIL Image): Image to be cropped.
            mask_box: the mask region, the cropped 2 views should contain this region. (x,y,w,h)

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a discrete crop view1
            tuple: params (i, j, h, w) to be passed to ``crop`` for a discrete crop view2
                Note: x=i, y=j is the coordinates of top left corner
            torch.tensor: mask map for view1
            torch.tensor: mask map for view2
        """
        box_x = mask_box[0]  # [0 ~ img_w - 1]
        box_y = mask_box[1]
        box_w = mask_box[2]
        box_h = mask_box[3]
        img_w = img.size[0] // self.patch_size  # 25 by default
        img_h = img.size[1] // self.patch_size

        pre_mask_map = np.zeros(shape=(img_w, img_h), dtype=int)
        for i in range(box_w):
            for j in range(box_h):
                pre_mask_map[box_y + j, box_x + i] = 1

        crop_w = self.size[0] // self.patch_size  # 14 by default
        crop_h = self.size[1] // self.patch_size
        mask_map = np.zeros(shape=(crop_w, crop_h), dtype=int)

        crop_i = random.randint(max(0, box_x + box_w - crop_w), min(box_x, img_w - crop_w))
        crop_j = random.randint(max(0, box_y + box_h - crop_h), min(box_y, img_h - crop_h))
        flip = random.random()
        flip = True if flip > 0.5 else False
        # crop and flip the mask map
        if flip:
            for i in range(crop_w):
                for j in range(crop_h):
                    mask_map[j, crop_w - i - 1] = pre_mask_map[j + crop_j, i + crop_i]
        else:
            for i in range(crop_w):
                for j in range(crop_h):
                    mask_map[j, i] = pre_mask_map[j + crop_j, i + crop_i]
        return crop_i * self.patch_size, crop_j * self.patch_size, \
               crop_w * self.patch_size, crop_h * self.patch_size, flip, mask_map

    def __call__(self, img, mask_box):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
            mask_box: the mask region: (x, y, w, h)  token-wise
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i1, j1, w1, h1, flip1, mask_map_q = self.get_params(img, mask_box)
        i2, j2, w2, h2, flip2, mask_map_k = self.get_params(img, mask_box)

        img_q = F.crop(img=img, top=j1, left=i1, height=h1, width=w1)
        img_k = F.crop(img=img, top=j2, left=i2, height=h2, width=w2)
        if flip1:
            img_q = F.hflip(img_q)
        if flip2:
            img_k = F.hflip(img_k)
        return img_q, img_k, mask_map_q, mask_map_k

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}, patch_size={1}'.format(self.size, self.patch_size)
        format_string += ')'
        return format_string


class RandomResizedCropAndFlipBothTwoImg:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation with random flip
    two images are of the same shape, and will be processed in exactly the same way

    This is to process two images simultaneously (original image and its importance/mask map)

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, img_2=None):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if img_2 is None:
            img = F.resized_crop(img, i, j, h, w, self.size, interpolation)
            if torch.rand(1) < 0.5:
                return F.hflip(img)
            return img
        else:
            img = F.resized_crop(img, i, j, h, w, self.size, interpolation)
            img_2 = F.resized_crop(img_2, i, j, h, w, self.size, interpolation)
            if torch.rand(1) < 0.5:
                return F.hflip(img), F.hflip(img_2)
            return img, img_2

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ')'
        return format_string
