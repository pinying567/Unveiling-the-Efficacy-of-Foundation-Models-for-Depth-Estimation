# Ref: https://github.com/shariqfarooq123/AdaBins/blob/main/evaluate.py

import torch


def compute_errors(pred, gt):

    batch_size = gt.size(0)
    valid_mask = (gt > 1e-3) & (gt < 10) & (pred > 1e-3)

    a1, a2, a3, = 0, 0, 0
    abs_rel, sq_rel, rmse_sum, rmse_log_sum = 0, 0, 0, 0
    silog, log_10 = 0, 0

    for i in range(batch_size):
        gt_valid = gt[i, valid_mask[i]].clamp(1e-3, 10)
        pred_valid = pred[i, valid_mask[i]].clamp(1e-3, 10)

        thresh = torch.max((gt_valid / pred_valid), (pred_valid / gt_valid))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_rel += torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
        sq_rel += torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)

        rmse = (gt_valid - pred_valid) ** 2
        rmse_sum += torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt_valid) - torch.log(pred_valid)) ** 2
        rmse_log_sum += torch.sqrt(rmse_log.mean())

        err = torch.log(pred_valid) - torch.log(gt_valid)
        silog += torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

        log_10 += (torch.abs(torch.log10(gt_valid) - torch.log10(pred_valid))).mean()

    a1, a2, a3 = a1 / batch_size, a2 / batch_size, a3 / batch_size
    abs_rel, sq_rel = abs_rel / batch_size, sq_rel / batch_size
    rmse_sum, rmse_log_sum = rmse_sum / batch_size, rmse_log_sum / batch_size
    silog, log_10 = silog / batch_size, log_10 / batch_size

    return dict(a1=a1.item(), a2=a2.item(), a3=a3.item(), abs_rel=abs_rel.item(), rmse=rmse_sum.item(), log_10=log_10.item(), rmse_log=rmse_log_sum.item(), silog=silog.item(), sq_rel=sq_rel.item())


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeters(object):
    def __init__(self, metrics):
        self.meters = {}
        for m in metrics:
            self.meters[m] = averageMeter()

    def update(self, new_dict, n=1):
        for m, value in new_dict.items():
            self.meters[m].update(value, n)

    def get_values(self):
        return {m: x.avg for m, x in self.meters.items()}


