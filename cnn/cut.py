#usage: cut.py <path_to_folder_to_cut>
#ex: cut.py dataset.001_sorted_augmented

import sys
import os
import cv2
import math

DST_DIR = 'cutted'

def cut(root, filename):
    path_pieces = root.split(os.sep)
    class_ = path_pieces[-1]
    viborka = path_pieces[-2]

    dst_path = os.path.join(DST_DIR, viborka, class_)
    os.makedirs(dst_path, exist_ok=True)

    src_path = os.path.join(root, filename)

    img = cv2.imread(src_path)

    height, width, channels = img.shape

    dh = math.floor(height / 10)
    dw = math.floor(width / 10)

    cur_h = 0
    cur_w = 0

    for i in range(10):
        for j in range(10):
            fname_pieces = filename.split('.')
            number = i * 10 + j
            new_fname = fname_pieces[0] + "_" + str(number) + "." + fname_pieces[-1]

            tmp_img = img[cur_h:cur_h+dh, cur_w:cur_w+dw].copy()
            cur_h += dh

            new_path = os.path.join(dst_path, new_fname)
            print(new_path)
            cv2.imwrite(new_path, tmp_img)
        cur_h = 0
        cur_w += dw

    # shutil.copy(os.path.join(root, filename), path)


def main(argv):
    os.makedirs(DST_DIR, exist_ok=True)

    for root, dirs, files in os.walk(argv[0]):
        for f in files:
            print(root, f)
            cut(root, f)

if __name__ == "__main__":
    main(sys.argv[1:])