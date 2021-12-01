import os
from scipy.io import loadmat
import shutil


def main():
    create_datasets(f'/home/ubuntu/xu')


def create_datasets(root):
    dataset_path = os.path.join(root, 'cars')

    cars = os.path.join(root, "cars2")
    trainF = os.path.join(cars, "train")
    valF = os.path.join(cars, "val")
    mkdir(cars)
    mkdir(trainF)
    mkdir(valF)

    train_list_path = os.path.join(dataset_path, 'devkit', 'cars_train_annos.mat')
    train_image_path = os.path.join(dataset_path, 'cars_train')

    test_list_path = os.path.join(dataset_path, 'cars_test_annos_withlabels.mat')
    test_image_path = os.path.join(dataset_path, 'cars_test')

    train_mat = loadmat(train_list_path)
    train_images = [f.item() for f in train_mat['annotations']['fname'][0]]
    train_labels = [f.item() for f in train_mat['annotations']['class'][0]]

    test_mat = loadmat(test_list_path)
    test_images = [f.item() for f in test_mat['annotations']['fname'][0]]
    test_labels = [f.item() for f in test_mat['annotations']['class'][0]]

    # print()

    for img, label in zip(train_images, train_labels):
        ori = os.path.join(train_image_path, img)
        toT = os.path.join(trainF, str(label))
        print('from {} to {}'.format(ori, toT))
        mkdir(toT)
        shutil.copy(ori, toT)

    for img, label in zip(test_images, test_labels):
        ori = os.path.join(test_image_path, img)
        toT = os.path.join(valF, str(label))
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
