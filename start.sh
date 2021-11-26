#nnictl create --config config_nni.yml

python3 train.py -c main/ctfg_cub.yml \
  --model ctfg_14_7x2_384 --data_dir /hy-nas/CUB2 \
  --is_con_los \
  --is_changeSize \
  --pretrained_dir /hy-tmp/ctfg_fork/output/train/20211125-214325-ctfg_14_7x2_224-224/model_best.pth.tar

#python3 -m torch.distributed.launch --nproc_per_node=2 \
#  train.py -c main/ctfg_cub.yml \
#  --model ctfg_14_7x2_384 --data_dir /home/ubuntu/xu/CUB2 \
#  --pretrained_dir /home/ubuntu/xu/cct_14_7x2_384_imagenet.pth \
#  --is_con_loss #--apex-amp
