python3 -m torch.distributed.launch --nproc_per_node=2 \
  train.py -c main/ctfg_cub.yml \
  --model ctfg_14_7x2_384 --data_dir /home/ubuntu/Datas/CUB/CUB2 \
  --pretrained --pretrained_dir /home/ubuntu/Datas/cct_14_7x2_384_imagenet.pth

