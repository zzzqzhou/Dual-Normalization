import os
import argparse
import numpy as np
import medpy.metric.binary as mmb

from tqdm import tqdm
from PIL import Image
from model.unetdsbn import Unet2D
from utils.palette import color_map
from datasets.dataset import Dataset, ToTensor, CreateOnehotLabel

import torch
import torchvision.transforms as tfs
from torch.nn import DataParallel
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader
import nibabel as nib
from preprocess_func import norm
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./BraTS_2018/test')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--test_domain_list', nargs='+', type=str)
parser.add_argument('--model_dir', type=str,  default='./results/unet_dn_t2/model', help='model_dir')
parser.add_argument('--batch_size', type=int,  default=32)
parser.add_argument('--gpu_ids', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

def get_bn_statis(model, domain_id):
    means = []
    vars = []
    for name, param in model.state_dict().items():
        if 'bns.{}.running_mean'.format(domain_id) in name:
            means.append(param.clone())
        elif 'bns.{}.running_var'.format(domain_id) in name:
            vars.append(param.clone())
    return means, vars


def cal_distance(means_1, means_2, vars_1, vars_2):
    pdist = PairwiseDistance(p=2)
    dis = 0
    for (mean_1, mean_2, var_1, var_2) in zip(means_1, means_2, vars_1, vars_2):
        dis += (pdist(mean_1.reshape(1, mean_1.shape[0]), mean_2.reshape(1, mean_2.shape[0])) + pdist(var_1.reshape(1, var_1.shape[0]), var_2.reshape(1, var_2.shape[0])))
    return dis.item()



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("result.log"),
                        logging.StreamHandler()
                    ])
    domain_name_list = {'t1': '_t1.nii', 
                      't1ce': '_t1ce.nii', 
                      't2': '_t2.nii', 
                      'flair': '_flair.nii'}
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_ids
    model_dir = FLAGS.model_dir
    n_classes = FLAGS.n_classes
    test_domain_list = FLAGS.test_domain_list
    num_domain = len(test_domain_list)
    sample_list = os.listdir(FLAGS.data_dir)
    print('Start Testing.')
    
    for test_idx in range(num_domain):
        model = Unet2D(num_classes=n_classes, num_domains=2, norm='dsbn')
        model.load_state_dict(torch.load(os.path.join(model_dir, 'final_model.pth')))
        model = DataParallel(model).cuda()
        means_list = []
        vars_list = []
        for i in range(2):
            means, vars = get_bn_statis(model, i)
            means_list.append(means)
            vars_list.append(vars)

        model.train()
        total_dice = 0
        total_hd = 0
        total_asd = 0
        dice_list = []
        hd_list = []
        asd_list = []

        tbar = tqdm(sample_list)
        for idx, sample_name in enumerate(tbar):
            image_path = os.path.join(FLAGS.data_dir, sample_name, sample_name + domain_name_list[test_domain_list[test_idx]])
            mask_path = os.path.join(FLAGS.data_dir, sample_name, sample_name + '_seg.nii')

            nib_img = nib.load(image_path)
            nib_mask = nib.load(mask_path)

            image = nib_img.get_fdata()
            mask = nib_mask.get_fdata()
            mask[mask != 0] = 1
            pred_y = np.zeros(mask.shape)

            image = norm(image).astype(np.float32)
            
            with torch.no_grad():
                
                for ii in range(int(np.floor(image.shape[2] // FLAGS.batch_size))):
                    if (ii + 1) * FLAGS.batch_size < image.shape[2]:
                        vol = image[..., ii * FLAGS.batch_size : (ii + 1) * FLAGS.batch_size]
                    else:
                        vol = image[..., ii * FLAGS.batch_size:]                    
                    vol = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(1).cuda()
                    
                    dis = 99999999
                    best_out = None

                    for domain_id in range(2):
                        output = model(vol, domain_label=domain_id*torch.ones(vol.shape[0], dtype=torch.long))
                        means, vars = get_bn_statis(model, domain_id)
                        new_dis = cal_distance(means, means_list[domain_id], vars, vars_list[domain_id])
                        if new_dis < dis:
                            best_out = output
                            dis = new_dis
                    
                    output = best_out
                    pred = output.cpu().detach().numpy()
                    pred = np.argmax(pred, axis=1)
                    pred = np.transpose(pred, (1, 2, 0))
                    if (ii + 1) * FLAGS.batch_size < image.shape[2]:
                        pred_y[..., ii * FLAGS.batch_size : (ii + 1) * FLAGS.batch_size] = pred
                    else:
                        pred_y[..., ii * FLAGS.batch_size:] = pred

                dice = mmb.dc(pred_y, mask)
                total_dice += dice

                logging.info('Domain: {}, Sample {}, Sample Dice: {}, Average Dice: {}'.format(
                    test_domain_list[test_idx],
                    sample_name,
                    round(100 * dice, 2),
                    round(100 * total_dice / (idx + 1), 2)
                ))