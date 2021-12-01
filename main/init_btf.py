import os
import shutil


def create_datasets(root):
    dataset_path = os.path.join(root, 'Butterfly200')

    btf2 = os.path.join(root, "btf2")
    trainF = os.path.join(btf2, "train")
    valF = os.path.join(btf2, "val")
    mkdir(btf2)
    mkdir(trainF)
    mkdir(valF)

    train_img_txt = 'Butterfly200_train_release.txt'
    val_img_txt = 'Butterfly200_val_release.txt'

    train_file = open(os.path.join(dataset_path, train_img_txt))
    val_file = open(os.path.join(dataset_path, val_img_txt))

    for p in train_file:
        file_name = p.split('/')[0]
        pic_name = p.split('/')[1].split(' ')[0]
        ori = os.path.join(dataset_path, 'images_small', file_name, pic_name)
        toT = os.path.join(trainF, file_name)
        print('from {} to {}'.format(ori, toT))
        mkdir(toT)
        shutil.copy(ori, toT)

    for p in val_file:
        file_name = p.split('/')[0]
        pic_name = p.split('/')[1].split(' ')[0]
        ori = os.path.join(dataset_path, 'images_small', file_name, pic_name)
        toT = os.path.join(valF, file_name)
        print('from {} to {}'.format(ori, toT))
        mkdir(toT)
        shutil.copy(ori, toT)


def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


def main():
    create_datasets(f'D:\Project\Datas\\butterfly')


if __name__ == "__main__":
    main()
