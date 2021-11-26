#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='8VqrRBPH'
export NNI_SYS_DIR='/home/lirunze/xh/project/pytorch-image-models/output/nni/8VqrRBPH/trials/D5H8h'
export NNI_TRIAL_JOB_ID='D5H8h'
export NNI_OUTPUT_DIR='/home/lirunze/xh/project/pytorch-image-models/output/nni/8VqrRBPH/trials/D5H8h'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/home/lirunze/xh/project/pytorch-image-models'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python3 train.py -c main/ctfg_cub.yml --model ctfg_14_7x2_384 --data_dir /home/lirunze/xh/datas/CUB2 --pretrained_dir /home/lirunze/xh/datas/cct_14_7x2_384_imagenet.pth --is_con_los --is_nni 1>/home/lirunze/xh/project/pytorch-image-models/output/nni/8VqrRBPH/trials/D5H8h/stdout 2>/home/lirunze/xh/project/pytorch-image-models/output/nni/8VqrRBPH/trials/D5H8h/stderr
echo $? `date +%s%3N` >'/home/lirunze/xh/project/pytorch-image-models/output/nni/8VqrRBPH/trials/D5H8h/.nni/state'