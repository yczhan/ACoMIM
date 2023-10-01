#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
echo 'The current dir is: ' $CURDIR
# /ghome/zhanyc2/beit/object_detection

ARCH=$1
JOB_ID=$2
PRETRAINED=$3
GPUS=$4

PORT=${PORT:-29500}

WORK_DIR=$(echo ${CURDIR%.*} | sed -e "s/ghome/gdata/g" | sed -e "s/code/output/g" | sed -e "s/object_detection/$JOB_ID/g")/det/
echo 'The work dir is: ' $WORK_DIR


# if [ -z $WEIGHT_FILE ]; then
WEIGHT_FILE=$WORK_DIR/checkpoint_teacher.pth
$PYTHON $CURDIR/extract_backbone_weights.py $PRETRAINED $WEIGHT_FILE --checkpoint_key student && \
# fi

MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $CURDIR/train.py \
    $CURDIR/configs/cascade_rcnn/${ARCH}_maskrcnn_giou_4conv1f_coco_3x.py \
    --work-dir $WORK_DIR  --deterministic \
    --cfg-options model.backbone.use_checkpoint=True \
    model.pretrained=$WEIGHT_FILE \
    ${@:5}  && \

MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $CURDIR/test.py \
    $CURDIR/configs/cascade_rcnn/${ARCH}_maskrcnn_giou_4conv1f_coco_3x.py \
    $WORK_DIR/latest.pth \
    --eval bbox segm \
    --cfg-options model.backbone.use_checkpoint=True \
    ${@:5}

