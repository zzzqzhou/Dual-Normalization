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
import torch.nn as nn
import torchvision.transforms as tfs
from torch.nn import DataParallel
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/brats/npz_data')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--test_domain_list', nargs='+', type=str)
parser.add_argument('--model_dir', type=str,  default='./results/unet_dn/model', help='model_dir')
parser.add_argument('--batch_size', type=int,  default=32)
parser.add_argument('--save_label', dest='save_label', action='store_true')
parser.add_argument('--label_dir', type=str,  default='./results/unet_dn', help='model_dir')
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
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_ids
    model_dir = FLAGS.model_dir
    n_classes = FLAGS.n_classes
    test_domain_list = FLAGS.test_domain_list
    num_domain = len(test_domain_list)
    print('Start Testing.')

    cmap = color_map(n_color=256, normalized=False).reshape(-1)
    
    if not os.path.exists(FLAGS.label_dir):
        os.mkdir(FLAGS.label_dir)
    
    for test_idx in range(num_domain):
        # Load trained model parameters
        # Add an auxiliary BN layer to compute target BN statistics
        model = Unet2D(num_classes=n_classes, num_domains=3, norm='dsbn')
        trained_state_dict = torch.load(os.path.join(model_dir, 'final_model.pth'))
        model_state_dict = model.state_dict()
        state_dict = {k:v for k, v in trained_state_dict.items() if k in model_state_dict.keys()}
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

        model = DataParallel(model).cuda()
        means_list = []
        vars_list = []

        # Get running means and running vars of DN
        for i in range(2):
            means, vars = get_bn_statis(model, i)
            means_list.append(means)
            vars_list.append(vars)

        model.eval()
        
        # Set 'train' mode for computing target BN statistics and better results
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

        dataset = Dataset(base_dir=FLAGS.data_dir, split='test', domain_list=test_domain_list[test_idx],
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=FLAGS.n_classes),
                            ToTensor()
                        ]))
        dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        tbar = tqdm(dataloader, ncols=150)
        total_dice = 0
        total_hd = 0
        total_asd = 0
        dice_list = []
        hd_list = []
        asd_list = []
        with torch.no_grad():
            for idx, (batch, id) in enumerate(tbar):
                sample_data = batch['image'].cuda()
                onehot_mask = batch['onehot_label'].detach().numpy()
                mask = batch['label'].detach().numpy()
                dis = 99999999
                best_out = None

                # Get target BN statistics
                _ = model(sample_data, domain_label=2*torch.ones(sample_data.shape[0], dtype=torch.long))
                means, vars = get_bn_statis(model, 2)
                
                # Select best result
                for domain_id in range(2):
                    new_dis = cal_distance(means, means_list[domain_id], vars, vars_list[domain_id])
                    if new_dis < dis:
                        best_out = model(sample_data, domain_label=domain_id*torch.ones(sample_data.shape[0], dtype=torch.long))
                        dis = new_dis

                output = best_out
                pred_y = output.cpu().detach().numpy()
                pred_y = np.argmax(pred_y, axis=1)
                total_dice += mmb.dc(pred_y, mask)
                
                if pred_y.sum() == 0:
                    total_hd += 100
                    total_asd += 100
                else:
                    total_hd += mmb.hd95(pred_y, mask)
                    total_asd += mmb.asd(pred_y, mask)
                tbar.set_description('Domain: {}, Dice: {}, HD: {}, ASD: {}'.format(
                    test_domain_list[test_idx],
                    round(100 * total_dice / (idx + 1), 2),
                    round(total_hd / (idx + 1), 2),
                    round(total_asd / (idx + 1), 2)
                ))

                if FLAGS.save_label:
                    if not os.path.exists(os.path.join(FLAGS.label_dir, test_domain_list[test_idx])):
                        os.mkdir(os.path.join(FLAGS.label_dir, test_domain_list[test_idx]))
                    for i, pred_mask in enumerate(pred_y):
                        pred_mask = Image.fromarray(np.uint8(pred_mask.T))
                        pred_mask = pred_mask.convert('P')
                        pred_mask.putpalette(cmap)
                        pred_mask.save(os.path.join(FLAGS.label_dir, test_domain_list[test_idx], id[i] + '.png'))