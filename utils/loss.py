import torch
import torch.nn.functional as F

eps = 1e-8

def dice_loss(input, target, p=2, ignore_index=-100):
    n, c, h, w = input.size()
    prob = F.softmax(input, dim=1)
    prob_flatten = prob.permute(0, 2, 3, 1).contiguous().view(-1, c)

    target_flatten = target.view(n * h * w, 1)
    mask = target_flatten != ignore_index
    target_flatten = target_flatten[mask].view(-1, 1)

    prob_flatten = prob_flatten[mask.repeat(1, c)]
    prob_flatten = prob_flatten.contiguous().view(-1, c)

    target_one_hot = torch.scatter(torch.zeros_like(prob_flatten), 1, target_flatten, 1.0)
    prob_flatten = prob_flatten[:, 1:]
    target_one_hot = target_one_hot[:, 1:]
    dc = dice(prob_flatten, target_one_hot, p)
    return 1.0 - dc.mean()

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss

def dice_coef1(y, target):
    target = target.float()
    smooth = 1e-5

    interset = torch.sum(y * target)
    z_sum = torch.sum(y)
    y_sum = torch.sum(target)
    dice = (2.0 * interset + smooth) / (z_sum + y_sum + smooth)
    return dice.item()

def dice_coef2(y, target, num_classes=2):
    y = F.one_hot(y, num_classes).permute(0, 3, 1, 2)
    target = target.float()
    smooth = 1e-5

    total_dice = 0
    for i in range(num_classes):
        if i == 0:
            continue
        total_dice += dice_coef1(y[:, i, ...], target[:, i, ...])
    total_dice /= (num_classes - 1)
    return total_dice

def dice(y, target, p=2):
    intersection = torch.sum(y * target, dim=0)
    union = y.pow(p).sum(0) + target.pow(p).sum(0)
    return 2 * intersection / (union + eps)

def dice_coef(y, target, p=2):
    esp = 1e-8
    interset = torch.sum(torch.mul(y, target))
    gt = y.pow(p).sum()
    pre = target.pow(p).sum()
    dice = 2.0 * interset / (pre + gt + esp)
    return dice.item()