import os
import shutil

dst = "/datasets/kurvachkin/dataset.001_sorted_augmented"


for d in os.listdir(dst):
    dst_d = os.path.join(dst, d)
    for cl in os.listdir(dst_d):
        dst_cl = os.path.join(dst_d, cl)
        print(dst)
        l = sorted(os.listdir(dst_cl))
        for i in range(len(l)):
            old_path = os.path.join(dst_cl,l[i])
            new_path = os.path.join(dst_cl,'{:08d}.jpg'.format(i))
            # print(old_path,new_path)
            shutil.move(old_path,new_path)
