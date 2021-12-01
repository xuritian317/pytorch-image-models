import os
from scipy.io import loadmat
import shutil


def main():
    create_datasets(f'/home/ubuntu/xu')


def create_datasets(root):
    dataset_path = os.path.join(root, 'dogs')

    dogs2 = os.path.join(root, "dogs2")
    trainF = os.path.join(dogs2, "train")
    valF = os.path.join(dogs2, "val")
    mkdir(dogs2)
    mkdir(trainF)
    mkdir(valF)

    train_path = os.path.join(dataset_path, 'train_list.mat')
    test_path = os.path.join(dataset_path, 'test_list.mat')

    train_mat = loadmat(train_path)
    test_mat = loadmat(test_path)

    train_file = [f.item().item() for f in train_mat['file_list']]
    test_file = [f.item().item() for f in test_mat['file_list']]

    for p in train_file:
        file_name = p.split('/')[0]
        ori = os.path.join(dataset_path, 'Images', p)
        toT = os.path.join(trainF, file_name)
        print('from {} to {}'.format(ori, toT))
        # mkdir(toT)
        # shutil.copy(ori, toT)

    for p in test_file:
        file_name = p.split('/')[0]
        ori = os.path.join(dataset_path, 'Images', p)
        toT = os.path.join(valF, file_name)
        print('from {} to {}'.format(ori, toT))
        mkdir(toT)
        shutil.copy(ori, toT)


def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


if __name__ == "__main__":
    main()
