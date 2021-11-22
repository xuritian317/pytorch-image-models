python3 -m torch.distributed.launch --nproc_per_node=2 \
  train.py -c main/ctfg_cub.yml \
  --model ctfg_14_7x2_384 --data_dir /home/ubuntu/xu/CUB2 \
  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
  --is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/cct/cct_cub.yml \
#  --model cct_14_7x2_384 --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth