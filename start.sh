#nnictl create --config config_nni.yml

#python3 mytrain.py -c main/ctfg_cub_transfg.yml \
#  --model ctfg_14_7x2_384 --data_dir /hy-nas/CUB2 \
#  --is_con_los

#  --pretrained_dir /hy-nas/cct_14_7x2_384_imagenet.pth \

#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_nabirds.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/lirunze/xh/datas/nabirds2 \
#  --pretrained_dir /home/lirunze/xh/datas/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss

#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cub_transfg.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/lirunze/xh/datas/CUB2 \
#  --pretrained_dir /home/lirunze/xh/datas/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cub_ctfg.yml \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss --is_need_da

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cub_transfg.yml \
#  --model transfg_1472 \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/ViT-B_16.npz \
#  --is_con_loss #--apex-amp --is_ori_load

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cub_vit.yml \
#  --model vit_base_patch16_384 \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --apex-amp #--is_ori_load
#  --pretrained_dir /home/ubuntu/xu/ViT-B_16.npz \

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_imagenet.yml \
#  --data_dir /home/ubuntu/Projects/MobileViT-main/data/imagenet_data \
#  --is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cub_transfg.yml \
#  --model transfg_1472 \
#  --data_dir /home/ubuntu/xu/CUB2 --is_con_loss
#  --pretrained_dir /home/ubuntu/xu/ViT-B_16.npz \
#--is_con_loss #--is_ori_load

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cub_transfg.yml \
#  --model transfg_1472 \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained \
#  --pretrained_dir /home/ubuntu/xu/ViT-B_16.npz \
#  --is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cars.yml \
#  --data_dir /home/ubuntu/xu/cars2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
#  --is_need_da #--is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_dogs.yml \
#  --data_dir /home/ubuntu/Datas/dogs2 \
#  --pretrained_dir /home/ubuntu/Datas/cct_14_7x2_384_imagenet.pth \
#  --log-wandb --experiment ctfg \
#  --is_need_da #--is_con_loss

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_dogs.yml \
#  --model ctfg_14_7x2_384 \
#  --data_dir /home/ubuntu/Datas/dogs2 \
#  --pretrained_dir /home/ubuntu/Datas/cct_14_7x2_384_imagenet.pth \
#  --is_need_da #--is_con_loss \
##  --log-wandb --experiment ctfg \

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  mytrain.py -c main/ctfg_cub_ctfg.yml \
#  --model ctfg_14_7x2_448_no_conv --img-size 448 --batch-size 4 \
#  --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
#  --is_need_da --is_con_loss

#2021年12月10日10:07:44 ubuntu 241
python3 -m torch.distributed.launch --nproc_per_node=2 \
  mytrain.py -c main/ctfg_cub_ctfg.yml \
  --model ctfg_14_7x2_384 --img-size 384 --batch-size 16 \
  --data_dir /home/ubuntu/xu/cub2 \
  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
  --is_need_da --is_con_loss
