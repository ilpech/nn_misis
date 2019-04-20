import os
import shutil
import pathlib
import random

src_path = '/datasets/kurvachkin/dataset.001_equal'
dataset_path = '/datasets/kurvachkin/dataset.001_sorted'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

print(dataset_path)
random.seed(420)

try:
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)
except PermissionError:
    print('Unable to create directory. Permission denied')

for im_class in os.listdir(src_path):
    train_class_path = os.path.join(train_path, im_class)
    val_class_path = os.path.join(val_path, im_class)
    test_class_path = os.path.join(test_path, im_class)
    try:
        pathlib.Path(train_class_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(val_class_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(test_class_path).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print('Unable to create directory. Permission denied')

    class_path = os.path.join(src_path, im_class)
    samples_count = len(os.listdir(class_path))
    to_train_count = int(samples_count * 0.9)
    to_test_count = int(samples_count * 0.05)
    to_val_count = int(samples_count * 0.15)
    ls = os.listdir(class_path)
    print(to_val_count, to_test_count, to_train_count)

    random.shuffle(ls)
    sample_number = 0
    for sample in ls:
        filePath = os.path.join(class_path, os.path.basename(sample))
        if sample_number < to_train_count:
            shutil.copy(filePath, train_class_path)
            print('file {} cp to dir {}'.format(filePath, train_class_path))
        elif sample_number < to_train_count + to_test_count:
            shutil.copy(filePath, test_class_path)
            print('file {} cp to dir {}'.format(filePath, test_class_path))
        elif sample_number < to_train_count + to_test_count + to_val_count:
            shutil.copy(filePath, val_class_path)
            print('file {} cp to dir {}'.format(filePath, val_class_path))
        sample_number += 1
