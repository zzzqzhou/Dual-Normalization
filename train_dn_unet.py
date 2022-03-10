import os
import random
import datetime
import argparse
import numpy as np

from tqdm import tqdm
from model.unetdsbn import Unet2D
from utils.loss import dice_loss1
from datasets.dataset import Dataset, ToTensor, CreateOnehotLabel

import torch
import torchvision.transforms as tfs
from torch import optim
from torch.optim import Adam
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser('Dual Normalization U-Net Training')
parser.add_argument('--data_dir', type=str, default='./data/brats/npz_data')
parser.add_argument('--train_domain_list_1', nargs='+')
parser.add_argument('--train_domain_list_2', nargs='+')
parser.add_argument('--result_dir', type=str, default='./results/unet_dn')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--save_step', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--deterministic', dest='deterministic', action='store_true')
args = parser.parse_args()

def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

if __name__== '__main__':
    start_time = datetime.datetime.now()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    base_dir = args.data_dir
    batch_size = args.batch_size
    save_step = args.save_step
    lr = args.lr
    train_domain_list_1 = args.train_domain_list_1
    train_domain_list_2 = args.train_domain_list_2
    max_epoch = args.n_epochs
    result_dir = args.result_dir
    n_classes = args.n_classes
    log_dir = os.path.join(result_dir, 'log')
    model_dir = os.path.join(result_dir, 'model')

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataloader_train = []
    model = Unet2D(num_classes=n_classes, norm='dsbn', num_domains=2)
    params_num = sum(p.numel() for p in model.parameters())
    print("\nModle's Params: %.3fM" % (params_num / 1e6))
    model = DataParallel(model).cuda()

    optimizer = Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))

    exp_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    dataset_1 = Dataset(base_dir=base_dir, split='train', domain_list=train_domain_list_1, 
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=n_classes),
                            ToTensor()
                        ]))
    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    dataloader_train.append(dataloader_1)
    dataset_2 = Dataset(base_dir=base_dir, split='train', domain_list=train_domain_list_2, 
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=n_classes),
                            ToTensor()
                        ]))
    dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    dataloader_train.append(dataloader_2)
    
    for epoch_num in range(max_epoch):
        data_iter = [repeat_dataloader(dataloader_train[i]) for i in range(2)]
        print('Epoch: {}, LR: {}'.format(epoch_num, round(exp_lr.get_last_lr()[0], 6)))
        tbar = tqdm(dataloader_train[0], ncols=150)
        model.train()
        for i, batch in enumerate(tbar):

            ### get all domains' sample_batch ###
            sample_batches = [batch]
            other_sample_batches = [next(data_iter[i]) for i in range(1, 2)]
            sample_batches += other_sample_batches

            total_loss = 0
            count = 0
            for train_idx in range(2):
                count += 1
                sample_data, sample_label = sample_batches[train_idx]['image'].cuda(), sample_batches[train_idx]['onehot_label'].cuda()

                outputs_soft = model(sample_data, domain_label=train_idx*torch.ones(sample_data.shape[0], dtype=torch.long))
                loss = dice_loss1(outputs_soft, sample_label)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            tbar.set_description('Total Loss: {}'.format(round((total_loss / count), 6)))
        
        exp_lr.step()

        if (epoch_num + 1) % save_step == 0:
            model_save_model_path = os.path.join(model_dir, 'epoch_{}.pth'.format(epoch_num))
            torch.save(model.module.state_dict(), model_save_model_path)
            print('save model to {}'.format(model_save_model_path))
        
    model_save_model_path = os.path.join(model_dir, 'final_model.pth'.format(epoch_num))
    torch.save(model.module.state_dict(), model_save_model_path)
    print('save model to {}'.format(model_save_model_path))
    
    end_time = datetime.datetime.now()
    print('Finish running. Cost total time: {} hours'.format((end_time - start_time).seconds / 3600))