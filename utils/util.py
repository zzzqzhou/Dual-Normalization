import numpy as np
from scipy.ndimage import _ni_support
from medpy import metric
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,generate_binary_structure
from scipy import ndimage
import scipy
from medpy import metric
import torch


def transfer_model(pretrained_file, model):
    '''
    只导入pretrained_file部分模型参数
    tensor([-0.7119,  0.0688, -1.7247, -1.7182, -1.2161, -0.7323, -2.1065, -0.5433,-1.5893, -0.5562]
    update:
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
    :param pretrained_file:
    :param model:
    :return:
    '''
    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model
 
 
def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict:
    :param model_dict:
    :return:
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        # else:
        #     print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def _eval_dice(gt_y, pred_y):
    return 2 * np.sum(gt_y * pred_y) / (np.sum(gt_y) + np.sum(pred_y))

def _eval_dice2(gt_y, pred_y):
    num_classes = gt_y.shape[1]
    onehot_pred = np.zeros((pred_y.shape[0], num_classes, pred_y.shape[1], pred_y.shape[2]), dtype=np.float32)
    for i in range(num_classes):
        onehot_pred[:, i, ...] = (pred_y == i).astype(np.float32)
    dice_list = []
    total_dice = 0
    for i in range(num_classes):
        if i == 0:
            continue
        dice_coef = 2 * np.sum(gt_y[:, i, ...] * onehot_pred[:, i, ...]) / (np.sum(gt_y[:, i, ...]) + np.sum(onehot_pred[:, i, ...]))
        dice_list.append(dice_coef)
        total_dice += dice_coef
    total_dice /= (num_classes - 1)
    return total_dice, dice_list

def _connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def _eval_average_surface_distances(reference, result, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    return metric.binary.asd(result, reference)


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds


def asd(result, reference, voxelspacing=None, connectivity=1):
  
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


def calculate_hausdorff(lP,lT):
    return scipy.spatial.distance.directed_hausdorff(lP, lT)
    # return asd(lP, lT, spacing)


def _eval_haus(pred_y, gt_y):
    return metric.binary.hd95(gt_y, pred_y)