#nnictl create --config config_nni.yml

python3 train.py -c main/ctfg_cars.yml \
  --model ctfg_14_7x2_384 --data_dir /hy-nas/cars2 \
  --pretrained_dir /hy-nas/cct_14_7x2_384_imagenet.pth \
  --is_con_los

#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_nabirds.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/lirunze/xh/datas/nabirds2 \
#  --pretrained_dir /home/lirunze/xh/datas/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss --apex-amp

#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/lirunze/xh/datas/CUB2 \
#  --pretrained_dir /home/lirunze/xh/datas/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_dogs.yml \
#  --data_dir /home/ubuntu/xu/dogs2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss

