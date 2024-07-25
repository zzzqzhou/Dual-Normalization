# Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization
by [Ziqi Zhou](https://zzzqzhou.github.io/), [Lei Qi](http://palm.seu.edu.cn/qilei/), [Xin Yang](https://xy0806.github.io/), Dong Ni, [Yinghuan Shi](https://cs.nju.edu.cn/shiyh/index.htm). 

## Introduction

This repository is for our CVPR 2022 paper '[Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization](https://arxiv.org/abs/2112.11177)'. 


![](./picture/cvpr22_dn.PNG)

## Data Preparation

### Dataset
[BraTS 2018](https://www.med.upenn.edu/sbia/brats2018/data.html) | [MMWHS](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/) | [Abdominal-MRI](https://chaos.grand-challenge.org/) | [Abdominal-CT](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

### File Organization

T2 as source domain
``` 
├── [Your BraTS2018 Path]
    ├── nii_data
        ├── test
            ├── Brats18_2013_4_1
                ├── Brats18_2013_4_1_flair.nii
                ├── Brats18_2013_4_1_t1.nii
                ├── Brats18_2013_4_1_t1ce.ni
                ├── Brats18_2013_4_1_t2.nii
                └── Brats18_2013_4_1_seg.nii
            ...
        └── train
            ...
    ├── npz_data
        ├── train
            ├── t2_ss
                ├── sample0.npz, sample1.npz, xxx
            └── t2_sd
        └── test
            ├── t1
                ├── test_sample0.npz, test_sample1.npz, xxx
            ├── flair
            ├── t2
            └── t1ce
```

For Brats dataset, we combined the HGG and LGG (totally 285 cases) then random selected 80% cases as training set (228 cases) and rest 20% cases as testing set (57 cases). Due to the access limit to my former university's GPU server, I can't provide the original dataset spliting. I provide a new random dataset spliting (in "Brats_trian.list" and "Brats_test.list") following our original setting.

## Training and Testing

Train on source domain T2.
```
python -W ignore train_dn_unet.py \
  --train_domain_list_1 t2_ss --train_domain_list_2 t2_sd --n_classes 2 \
  --batch_size 128 --n_epochs 50 --save_step 10 --lr 0.004 --gpu_ids 0,1 \
  --result_dir ./results/unet_dn_t2 --data_dir [Your BraTS2018 Npz Training Data Folder]
```

Test on target domains (T1, T1ce and Flair).

```
python -W ignore test_dn_unet.py \
  --data_dir [Your BraTS2018 Npz Test Data Folfer] --n_classes 2 \
  --test_domain_list flair t1 t1ce --model_dir ./results/unet_dn_t2/model \
  --batch_size 64 --gpu_ids 0 \
```

## Acknowledgement
The U-Net model is borrowed from [Fed-DG](https://github.com/liuquande/FedDG-ELCFS). The Style Augmentation (SA) module is based on the nonlinear transformation in [Models Genesis](https://github.com/MrGiovanni/ModelsGenesis). The Dual-Normalizaiton is borrow from [DSBN](https://github.com/wgchang/DSBN). We thank all of them for their great contributions.

## Citation

If you find this project useful for your research, please consider citing:

```bibtex
@inproceedings{zhou2022dn,
  title={Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization},
  author={Zhou, Ziqi and Qi, Lei and Yang, Xin and Ni, Dong and Shi, Yinghuan},
  booktitle={CVPR},
  year={2022}
}
```
