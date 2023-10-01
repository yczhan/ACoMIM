#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG_PT=$1
CFG_FT=$2
GPUS=$3
PY_ARGS=${@:4}
PORT=${PORT:-29500}

WORK_DIR_PT=$(echo ${CFG_PT%.*} | sed -e "s/code/output/g" | sed -e "s/configs//g")/
WORK_DIR_FT=$(echo ${CFG_FT%.*} | sed -e "s/code/output/g" | sed -e "s/configs//g")/
echo Pretrain directory:
echo $WORK_DIR_PT
echo Finetune directory:
echo $WORK_DIR_FT

MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    /code/ACoMIM/run_beit_contrast.py $CFG_PT --output_dir $WORK_DIR_PT  &&\
MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    /code/ACoMIM/run_class_finetuning.py $CFG_FT --output_dir $WORK_DIR_FT --finetune $WORK_DIR_PT/checkpoint-latest.pth









