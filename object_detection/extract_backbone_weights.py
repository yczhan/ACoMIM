# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
import os


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    parser.add_argument("--checkpoint_key", default="state_dict", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--with_head", type=bool_flag, default=False, help='extract checkpoints w/ or w/o head")')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    file_dir = args.output.split('/checkpoint')[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    checkpoint_model = ck['model']
    output_dict = dict(state_dict=dict())
    has_backbone = False
    if args.checkpoint_key == 'teacher':
        print('\nload from the teacher network')
        for key, value in checkpoint_model.items():
            if key.startswith('blocks_k'):
                output_dict['state_dict'][key[:6] + key[8:]] = value  # blocks_k -> blocks
                has_backbone = True
            elif key.startswith('patch_embed_k'):
                output_dict['state_dict'][key[:11] + key[13:]] = value  # patch_embed_k -> patch_embed
                has_backbone = True
            elif key.startswith('blocks') or key.startswith('patch_embed'):
                pass
            else:
                output_dict['state_dict'][key] = value
    else:
        print('\nload from the student network')
        for key, value in checkpoint_model.items():
            if key.startswith('blocks_k') or key.startswith('patch_embed_k'):
                pass
            elif key.startswith('encoder'):
                output_dict['state_dict'][key[8:]] = value
                has_backbone = True
            else:
                output_dict['state_dict'][key] = value
                has_backbone = True
    if not has_backbone:
        # raise Exception("Cannot find a backbone module in the checkpoint.")
        print("Cannot find a backbone module in the checkpoint. No modification.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()