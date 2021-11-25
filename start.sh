#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss #--apex-amp

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/cct/cct_cub.yml \
#  --model cct_14_7x2_384 --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth


#python3 train.py -c main/ctfg_cub.yml \
#  --model ctfg_14_7x2_384 --data_dir /hy-nas/CUB2 \
#  --pretrained_dir /hy-nas/cct_14_7x2_384_imagenet.pth \
#  --is_con_los

nnictl create --config config_nni.yml