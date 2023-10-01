# ACoMIM
"ATTENTION-GUIDED CONTRASTIVE MASKED IMAGE MODELING FOR TRANSFORMER-BASED SELF-SUPERVISED LEARNING" ICIP 2023

Our codes & required packages are based on BEiT, Thanks!

## Example: Pre-training and fine-tuning on ImageNet-1K

```
bash /code/ACoMIM/tools/dist_ptft.sh /code/ACoMIM/configs/selfsup/beit/vitb_M3V3attnbox300.py /code/ACoMIM/configs/finetune_cls/beit/IN_vanilla_vitb_abs_ft.py 8

```


