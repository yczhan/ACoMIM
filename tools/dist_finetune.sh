#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
CKPT=$2
GPUS=$3
PY_ARGS=${@:4}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/code/output/g" | sed -e "s/configs//g")/
echo $WORK_DIR

MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    /code/ACoMIM/run_class_finetuning.py $CFG --output_dir $WORK_DIR --finetune $CKPT

# --enable_deepspeed













