import os
import json
from os.path import join
import shutil


def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


def copy(root):
    target = os.path.join(root, "CUB_200_2011")
    cub2 = os.path.join(root, "CUB2")
    trainF = os.path.join(cub2, "train")
    valF = os.path.join(cub2, "val")
    mkdir(cub2)
    mkdir(trainF)
    mkdir(valF)

    img_txt_file = open(os.path.join(target, 'images.txt'))
    train_val_file = open(os.path.join(target, 'train_test_split.txt'))

    img_name_list = []
    for line in img_txt_file:
        img_name_list.append(line[:-1].split(' ')[-1])

    train_test_list = []
    for line in train_val_file:
        train_test_list.append(int(line[:-1].split(' ')[-1]))

    for i, x in zip(train_test_list, img_name_list):
        if i:
            ori = os.path.join(target, 'images', x)
            name = x.split('/')[0]
            toT = os.path.join(trainF, name)
            mkdir(toT)
            shutil.copy(ori, toT)
        else:
            ori = os.path.join(target, 'images', x)
            name = x.split('/')[0]
            toT = os.path.join(valF, name)
            mkdir(toT)
            shutil.copy(ori, toT)


def main():
    copy(f'/home/ubuntu/xu')


if __name__ == '__main__':
    main()
