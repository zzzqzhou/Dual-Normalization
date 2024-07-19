import random
import os
import numpy as np
import shutil


if __name__ == "__main__":
    modality_name_list = {'t1': '_t1.nii.gz', 
                        't1ce': '_t1ce.nii.gz', 
                        't2': '_t2.nii.gz', 
                        'flair': '_flair.nii.gz'}
    src_dir_HGG = "./Brats_2018/HGG"
    src_dir_LGG = './Brats_2018/LGG'
    train_dir = "./BraTS_2018/train"
    test_dir = "./BraTS_2018/test"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    src_list = []
    src_list.extend([os.path.join(src_dir_HGG, item) for item in os.listdir(src_dir_HGG)] )
    src_list.extend([os.path.join(src_dir_LGG, item) for item in os.listdir(src_dir_LGG)] )

    random.shuffle(src_list)

    spilt_point = int(0.8 * len(src_list))

    train_list = src_list[:spilt_point]
    test_list = src_list[spilt_point:]

    for item in train_list:
        print(item)
        dest_path = os.path.join(train_dir, os.path.basename(item))
        shutil.copytree(item, dest_path)
    
    for item in test_list:
        dest_path = os.path.join(test_dir, os.path.basename(item))
        shutil.copytree(item, dest_path)