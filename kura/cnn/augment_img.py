import os
import sys
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
# https://github.com/albu/albumentations
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def strong_aug(p=1):
    return Compose([
        ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=2, p=0.5),
        OneOf([
            MotionBlur(p=.33),
            MedianBlur(blur_limit=3, p=.33),
            Blur(blur_limit=3, p=.33),
        ], p=0.25),
        HorizontalFlip(p=0.33),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.15),
    ], p=p)

augmentation = strong_aug(p=1.0)

data_path = "/datasets/kurvachkin/dataset.001_sorted/train"

for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)
    for img_name in os.listdir(class_path):
        if img_name == '.DS_Store':
            continue
        if len(os.listdir(class_path)) >= 40:
            break
        img_fname = os.path.join(class_path, img_name)
        img = cv2.imread(img_fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented_img = augmentation(image=img)['image']

        augm_img_fname = os.path.join(class_path, 'augm_1_' + img_name)

        matplotlib.image.imsave(augm_img_fname, augmented_img)
