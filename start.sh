#nnictl create --config config_nni.yml

#python3 train.py -c main/ctfg_cub_transfg.yml \
#  --model ctfg_14_7x2_384 --data_dir /hy-nas/CUB2 \
#  --is_con_los

#  --pretrained_dir /hy-nas/cct_14_7x2_384_imagenet.pth \

#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_nabirds.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/lirunze/xh/datas/nabirds2 \
#  --pretrained_dir /home/lirunze/xh/datas/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss

#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub_transfg.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/lirunze/xh/datas/CUB2 \
#  --pretrained_dir /home/lirunze/xh/datas/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub_ctfg.yml \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss --is_need_da

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub_transfg.yml \
#  --model transfg_1472 \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/ViT-B_16.npz \
#  --is_con_loss #--apex-amp --is_ori_load

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub_vit.yml \
#  --model vit_base_patch16_384 \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --apex-amp #--is_ori_load
#  --pretrained_dir /home/ubuntu/xu/ViT-B_16.npz \

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_imagenet.yml \
#  --data_dir /home/ubuntu/Projects/MobileViT-main/data/imagenet_data \
#  --is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub_transfg.yml \
#  --model transfg_1472 \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/ViT-B_16.npz \
#  --is_con_loss --is_ori_load

#python3 train.py -c main/ctfg_cub.yml \
#  --model transfg_1472 \
#  --data_dir /hy-nas/CUB2 \
#  --pretrained_dir /hy-nas/ViT-B_16.npz \
#  --is_con_loss --is_ori_load


python3 -m torch.distributed.launch --nproc_per_node=2 \
  train.py -c main/ctfg_cars.yml \
  --data_dir /home/ubuntu/xu/cars \
  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
  --is_con_loss --is_need_da