import os
import pandas as pd
import shutil


def create_datasets(root):
    dataset_path = os.path.join(root, 'nabirds')

    nab2 = os.path.join(root, "nabirds2")
    trainF = os.path.join(nab2, "train")
    valF = os.path.join(nab2, "val")
    mkdir(nab2)
    mkdir(trainF)
    mkdir(valF)

    image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                              sep=' ', names=['img_id', 'filepath'])
    # image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
    #                                  sep=' ', names=['img_id', 'target'])
    # Since the raw labels are non-continuous, map them to new ones
    # label_map = get_continuous_class_map(image_class_labels['target'])

    train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                   sep=' ', names=['img_id', 'is_training_img'])
    # data = image_paths.merge(image_class_labels, on='img_id')

    data = image_paths.merge(train_test_split, on='img_id')

    train_data = data[data.is_training_img == 1]
    test_data = data[data.is_training_img == 0]

    # class_names = load_class_names(dataset_path)
    # class_hierarchy = load_hierarchy(dataset_path)

    for p, i in zip(data['filepath'], data['is_training_img']):
        ori = os.path.join(dataset_path, 'images', p)
        name = p.split('/')[0]
        if i == 1:
            toT = os.path.join(trainF, name)
        else:
            toT = os.path.join(valF, name)
        mkdir(toT)
        shutil.copy(ori, toT)


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents


def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


def main():
    create_datasets(f'/home/lirunze/xh/datas')


if __name__ == "__main__":
    main()
