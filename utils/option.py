import argparse

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Seg Brats2018')
        parser.add_argument('--train_data', type=str, default='/data/ziqi/datasets/brats/npz_data_switch/train')
        parser.add_argument('--val_data', type=str, default='/data/ziqi/datasets/brats/npz_data_switch/val')
        parser.add_argument('--test_data', type=str, default='/data/ziqi/datasets/brats/npz_data_switch/test')
        parser.add_argument('--train_mode_1', type=str, default='t2')
        parser.add_argument('--train_mode_2', type=str, default='aug1')
        parser.add_argument('--train_mode_3', type=str, default='aug2')
        parser.add_argument('--train_mode_4', type=str, default='aug3')
        parser.add_argument('--train_mode_5', type=str, default='aug4')
        parser.add_argument('--train_mode_6', type=str, default='aug5')
        parser.add_argument('--test_mode', type=str, default='t1')
        parser.add_argument('--test_mode_1', type=str, default='t2')
        parser.add_argument('--test_mode_2', type=str, default='flair')
        parser.add_argument('--test_mode_3', type=str, default='t1ce')
        parser.add_argument('--test_mode_4', type=str, default='t1')
        parser.add_argument('--val_mode', type=str, default='t2')
        parser.add_argument('--result_dir', type=str, default='./results/resnet')
        parser.add_argument('--n_classes', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--val_batch_size', type=int, default=8)
        parser.add_argument('--test_batch_size', type=int, default=8)
        parser.add_argument('--n_epochs', type=int, default=120)
        parser.add_argument('--lr', type=float, default=0.002)
        parser.add_argument('--betas_1', type=float, default=0.9)
        parser.add_argument('--betas_2', type=float, default=0.999)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--backbone', type=str, default='resnet')
        parser.add_argument('--n_gpus', type=str, default='0, 1, 2, 3, 4, 5, 6, 7')
        parser.add_argument('--checkpoint', type=str, default='./results/resnet_psp/models/model_best.pth')
        parser.add_argument('--ckp_dir', type=str, default='./results/unet_multi/models')
        parser.add_argument('--save_mask', dest='save_mask', action='store_true')
        parser.add_argument('--save_label', dest='save_label', action='store_true')
        parser.add_argument('--random_weight', dest='random_weight', action='store_true')
        parser.add_argument('--output_ensemble', dest='output_ensemble', action='store_true')
        self.parser = parser
    
    def parse(self):
        args = self.parser.parse_args()
        # print(args)
        return args